"""Compare optimized execution to the published native models, not a copy of it."""

import copy
import math
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from jingwei_benchmark._native.models import DMF
from jingwei_benchmark.torch_engine import factorize_dmf, masked_loss


@pytest.mark.parametrize("hidden", [[], [7], [7, 5]])
def test_factorized_dmf_preserves_native_outputs_gradients_and_parameters(hidden):
    torch.manual_seed(21)
    native = DMF(5, 11, embedding_dim=3, hidden_dims=hidden, mask_predictor_hidden_dim=7)
    optimized = copy.deepcopy(native)
    assert factorize_dmf(optimized)
    assert list(native.state_dict()) == list(optimized.state_dict())
    # Repeated, shuffled sample IDs also exercise embedding gradient accumulation.
    rows = torch.tensor([4, 0, 4, 1])
    inputs = torch.randn(4, 11)
    torch.manual_seed(8)
    actual = native(inputs, row_indices=rows)
    sum(part.square().mean() for part in actual).backward()
    torch.manual_seed(8)
    expected = optimized(inputs, row_indices=rows)
    sum(part.square().mean() for part in expected).backward()
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
    for a, b in zip(native.parameters(), optimized.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=5e-5, atol=2e-6)


@pytest.mark.parametrize("method", ["dmf", "dcae"])
@pytest.mark.parametrize("weight,empty", [(0.5, False), (0.0, False), (0.5, True)])
def test_dense_mask_loss_and_adamw_step_match_native_training_step(method, weight, empty):
    from jingwei_benchmark._native.dmf import DMFImputer
    from jingwei_benchmark._native.dcae import DCAEImputer
    torch.set_num_threads(1)
    torch.manual_seed(4)
    target = torch.randn(4, 17)
    supervision = (torch.rand(4, 17) > .3).float() if not empty else torch.zeros_like(target)
    input_mask = supervision * (torch.rand(4, 17) > .2)
    target *= supervision
    common = dict(full_data_tensor=target, full_mask_tensor=input_mask, supervision_mask_tensor=supervision, batch_size=4)
    reference = (DMFImputer(**common, embedding_dim=3, hidden_dims=[7], mask_weight=weight)
                 if method == "dmf" else DCAEImputer(**common, ae_dim=4, mask_predictor_hidden_dim=7,
                                                    num_encoder_blocks=1, num_decoder_blocks=1, lambda_mask=weight))
    optimized = copy.deepcopy(reference.model)
    native_optimizer = reference.configure_optimizers()
    optimized_optimizer = torch.optim.AdamW(optimized.parameters(),
        lr=reference.learning_rate, weight_decay=reference.weight_decay if method == "dmf" else 1e-5)
    batch = (target * input_mask, target, input_mask, supervision)
    indices = torch.arange(4)
    torch.manual_seed(9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        native_loss = reference.training_step((*batch, indices) if method == "dmf" else batch, 0)
    native_loss.backward()
    torch.manual_seed(9)
    output, logits = (optimized(batch[0], row_indices=indices) if method == "dmf" else optimized(batch[0], input_mask))
    loss = masked_loss(output, logits, target, input_mask, supervision, weight)
    loss.backward()
    torch.testing.assert_close(loss, native_loss, rtol=2e-6, atol=2e-6)
    for actual, expected in zip(reference.model.parameters(), optimized.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=3e-5, atol=2e-6)
    native_optimizer.step()
    optimized_optimizer.step()
    for actual, expected in zip(reference.model.parameters(), optimized.parameters()):
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-6)


@pytest.mark.parametrize("name", ["jingwei-dmf", "jingwei-dcae"])
def test_torch_runner_retains_two_mask_protocol_and_complete_training_budget(name, tmp_path):
    from jingwei_benchmark.methods import run_method
    values = np.random.default_rng(42).normal(size=(7, 17))
    values[0, 0] = np.nan
    hidden = np.zeros(values.shape, dtype=bool)
    hidden[1, :3] = True
    parameters = dict(max_epochs=2, batch_size=3, latent_dim=4, hidden_dims=[7],
                      encoder_blocks=1, decoder_blocks=1, training_engine="torch")
    a, metadata = run_method(name, values, hidden, parameters, device="cpu", seed=7, folder=tmp_path)
    b, _ = run_method(name, values, hidden, parameters, device="cpu", seed=7, folder=tmp_path)
    np.testing.assert_array_equal(a[np.isfinite(values)], values[np.isfinite(values)])
    np.testing.assert_array_equal(a, b)
    assert np.isfinite(a).all()
    assert metadata["epochs_trained"] == 2
    assert metadata["training_steps"] == 2 * math.ceil(len(values) / 3)
    assert metadata["supervision_entries_per_epoch"] == np.isfinite(values).sum()
    assert metadata["training_input_entries"] == (np.isfinite(values) & ~hidden).sum()
    assert metadata["actual_device"] == "cpu"


@pytest.mark.parametrize("name", ["jingwei-dmf", "jingwei-dcae"])
@pytest.mark.parametrize("seed", [7, 31])
def test_native_and_lightning_engines_share_initialization_shuffle_and_imputation(name, seed, monkeypatch):
    import jingwei_benchmark._native.models as native_models
    from jingwei_benchmark.torch_engine import run_jingwei_torch
    from jingwei_benchmark.lightning_engine import run_jingwei_lightning
    torch.set_num_threads(1)
    model_type = native_models.DMF if name.endswith("dmf") else native_models.DCAE
    original_init = model_type.__init__
    instances = []

    def observe_init(model, *args, **kwargs):
        original_init(model, *args, **kwargs)
        observation = {"state": {key: tensor.clone() for key, tensor in model.state_dict().items()}, "batches": []}
        instances.append(observation)
        def capture(module, args, kwargs):
            if module.training:
                observation["batches"].append(args[0].detach().cpu().clone())
        model.register_forward_pre_hook(capture, with_kwargs=True)

    monkeypatch.setattr(model_type, "__init__", observe_init)
    values = np.random.default_rng(4).normal(size=(7, 17))
    values[0, 0] = np.nan  # First/evaluation-corrupted input; truth never enters either engine.
    hidden = np.zeros(values.shape, dtype=bool)
    hidden[1, :3] = True
    params = dict(max_epochs=2, batch_size=3, latent_dim=4, hidden_dims=[7], encoder_blocks=1, decoder_blocks=1)
    optimized, meta = run_jingwei_torch(name, values, hidden, params, device="cpu", seed=seed)
    native, reference_meta = run_jingwei_lightning(name, values, hidden, params, device="cpu", seed=seed)
    assert len(instances) == 2
    for key, tensor in instances[0]["state"].items():
        torch.testing.assert_close(tensor, instances[1]["state"][key], rtol=0, atol=0)
    assert len(instances[0]["batches"]) == len(instances[1]["batches"]) == meta["training_steps"]
    for a, b in zip(instances[0]["batches"], instances[1]["batches"]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert meta["training_steps"] == reference_meta["training_steps"]
    np.testing.assert_allclose(optimized, native, rtol=3e-5, atol=3e-5)
