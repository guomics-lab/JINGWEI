import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from jingwei import BenchmarkMasks, CSVDataset, DCAEImputer, DMFImputer
from jingwei.models import DMF

torch.set_num_threads(1)


@pytest.mark.parametrize('seed',[0,42])
def test_public_get_mask_matches_pinned_original_values_and_random_layout(tmp_path,seed):
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'software/benchmark/src/jingwei_benchmark/_legacy/datasets.py'
    spec=importlib.util.spec_from_file_location('original_mask_dataset',path)
    reference_module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_module)
    values=np.random.default_rng(4).normal(20,3,(48,9))
    values[0,2]=np.nan
    source=tmp_path/'data.csv'
    pd.DataFrame(values).to_csv(source)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reference=reference_module.CSVDataset(source)
    actual=CSVDataset(source,seed=seed)
    torch.testing.assert_close(actual.mean,reference.mean,rtol=0,atol=0)
    torch.testing.assert_close(actual.std,reference.std,rtol=0,atol=0)
    torch.testing.assert_close(actual.mask,reference.mask,rtol=0,atol=0)


def make_dataset(tmp_path, held_out_value):
    values = np.array(
        [[1.0, 3.0, 5.0, 7.0], [held_out_value, 5.0, 0.0, 9.0], [3.0, 7.0, 9.0, 11.0]]
    )
    observed = values > 0
    evaluation = np.zeros_like(observed)
    evaluation[1, 0] = True
    training = np.zeros_like(observed)
    training[0, 1] = True
    masks = BenchmarkMasks(observed, evaluation, training)
    filename = tmp_path / f"input_{held_out_value}.csv"
    pd.DataFrame(values, index=["A", "B", "C"]).to_csv(filename)
    return CSVDataset(filename, masks=masks)


def test_evaluation_truth_does_not_affect_normalization_or_training_data(tmp_path):
    first, changed_truth = make_dataset(tmp_path, 1000), make_dataset(tmp_path, 1000000)
    assert torch.equal(first.mean, changed_truth.mean)
    assert torch.equal(first.std, changed_truth.std)
    assert torch.equal(first.data_normalized, changed_truth.data_normalized)
    assert first.mean[0, 0] == 2
    assert first.std[0, 0] == 1
    assert first.data_normalized[1, 0] == 0
    assert first.supervision_mask[0, 1] == 1
    assert first.mask[0, 1] == 0
    assert first.supervision_mask[1, 0] == 0


class ConstantModel(nn.Module):
    def __init__(self, rows, features):
        super().__init__()
        self.output = nn.Parameter(torch.zeros(rows, features))
        self.logits = nn.Parameter(torch.zeros(rows, features))

    def forward(self, values, row_indices=None):
        return self.output, self.logits

    def get_full_matrix(self):
        return self.output


def make_imputer(kind, data, inputs, supervision):
    options = dict(
        full_data_tensor=data,
        full_mask_tensor=inputs,
        supervision_mask_tensor=supervision,
        batch_size=len(data),
    )
    if kind == "dmf":
        return DMFImputer(**options, embedding_dim=2, hidden_dims=[4], mask_weight=1)
    return DCAEImputer(
        **options,
        ae_dim=4,
        mask_predictor_hidden_dim=4,
        num_encoder_blocks=1,
        num_decoder_blocks=1,
        lambda_mask=1,
    )


@pytest.mark.parametrize("kind", ["dmf", "dcae"])
def test_training_loss_supervises_second_mask_and_excludes_evaluation(kind):
    data = torch.tensor([[2.0, 4.0, 1000.0, float("nan")]])
    supervision = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    inputs = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    model = make_imputer(kind, data, inputs, supervision)
    model.model = ConstantModel(1, 4)
    model.log = lambda *a, **k: None
    model.log_dict = lambda *a, **k: None
    batch = next(iter(model.train_dataloader()))
    assert batch[0][0, 1] == 0  # Second mask is hidden from the network input.
    assert batch[1][0, 1] == 4  # Yet remains a supervised reconstruction target.
    assert torch.isfinite(batch[1]).all()  # First-mask truths never reach the model.
    loss = model.training_step(batch, 0)
    assert loss.item() == pytest.approx(10 + np.log(2))
    loss.backward()
    assert model.model.output.grad[0, 1] != 0
    assert torch.equal(model.model.output.grad[0, 2:], torch.zeros(2))
    assert torch.equal(model.model.logits.grad[0, 2:], torch.zeros(2))


@pytest.mark.parametrize("kind", ["dmf", "dcae"])
def test_inference_restores_training_observations_and_imputes_test_cells(kind):
    data = torch.tensor([[2.0, 4.0, 1000.0, float("nan")]])
    supervision = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    inputs = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    model = make_imputer(kind, data, inputs, supervision)
    model.model = ConstantModel(1, 4)
    with torch.no_grad():
        model.model.output.fill_(9.0)
    assert torch.equal(model.get_imputed_data(), torch.tensor([[2.0, 4.0, 9.0, 9.0]]))


@pytest.mark.parametrize("kind", ["dmf", "dcae"])
def test_invalid_input_supervision_nesting_rejected(kind):
    with pytest.raises(ValueError, match="subset"):
        make_imputer(kind, torch.ones((1, 4)), torch.ones((1, 4)), torch.zeros((1, 4)))


def test_dmf_batch_row_identity_survives_shuffling():
    model = DMF(
        n_rows=5,
        n_cols=4,
        embedding_dim=2,
        hidden_dims=[4],
        mask_predictor_hidden_dim=4,
    )
    model.eval()
    row_indices = torch.tensor([4, 1, 3])
    values, _ = model(torch.zeros(3, 4), row_indices=row_indices)
    assert torch.allclose(values, model.get_full_matrix()[row_indices])


@pytest.mark.parametrize("kind", ["dmf", "dcae"])
def test_actual_model_training_and_inference_with_two_masks(kind, tmp_path):
    dataset = make_dataset(tmp_path, 1000)
    model = make_imputer(
        kind, dataset.data_normalized, dataset.mask, dataset.supervision_mask
    )
    model.log = lambda *a, **k: None
    model.log_dict = lambda *a, **k: None
    batch = next(iter(model.train_dataloader()))
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    result = model.get_imputed_data()
    assert torch.isfinite(result).all()
    available = dataset.supervision_mask.bool()
    assert torch.equal(result[available], dataset.data_normalized[available])


def test_valid_negative_values_can_be_declared_observed(tmp_path):
    filename = tmp_path / "log_data.csv"
    pd.DataFrame([[-2.0, 0.0, 3.0], [-1.0, 1.0, 4.0]]).to_csv(filename)
    dataset = CSVDataset(filename, missing_values="nan", seed=3)
    assert dataset.observation_mask.bool().all()


@pytest.mark.parametrize("kind", ["dmf", "dcae"])
def test_empty_supervision_batch_backpropagates_zero_without_nan(kind):
    data = torch.ones((2, 4))
    supervision = torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    model = make_imputer(kind, data, supervision, supervision)
    model.log = lambda *a, **k: None
    model.log_dict = lambda *a, **k: None
    batch = [torch.zeros((1, 4)) for _ in range(4)]
    if kind == "dmf":
        batch.append(torch.tensor([1]))
    loss = model.training_step(batch, 0)
    assert loss.item() == 0
    loss.backward()
