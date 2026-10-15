import random

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pimmslearn")

from jingwei_benchmark.pimms_engine import cached_loaders, _cached_dataset, run_pimms


@pytest.fixture
def frame():
    rng = np.random.default_rng(23)
    data = pd.DataFrame(rng.normal(10, 1, (16, 12)), index=[f"S{i}" for i in range(16)], columns=[f"F{i}" for i in range(12)])
    data.index.name, data.columns.name = "sample", "feature"
    missing = rng.random(data.shape) < .15
    hidden = (~missing) & (rng.random(data.shape) < .2)
    return data.mask(missing), hidden


@pytest.mark.parametrize("validation", [False, True])
def test_cached_rows_equal_native_and_do_not_consume_rng(frame, validation):
    from pimmslearn.io import datasets
    from pimmslearn.transform import VaepPipeline
    from pimmslearn.sklearn.ae_transformer import default_pipeline

    values, hidden = frame
    train = values.mask(hidden)
    transform = VaepPipeline(df_train=train, encode=default_pipeline, decode=["normalize"])
    base = datasets.DatasetWithTargetSpecifyTarget if validation else datasets.DatasetWithTarget
    kwargs = dict(df=train, transformer=transform)
    if validation:
        kwargs["targets"] = values.where(hidden)
    native = base(**kwargs)
    state = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    cached = _cached_dataset(base, validation=validation)(**kwargs)
    torch.testing.assert_close(torch.get_rng_state(), state[0], rtol=0, atol=0)
    assert random.getstate() == state[1]
    np.testing.assert_array_equal(np.random.get_state()[1], state[2][1])
    for index in (0, 7, 15, 7):
        for a, b in zip(native[index], cached[index]):
            torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)


def test_cached_loaders_restore_attributes_on_failure(frame):
    from pimmslearn.io import datasets
    from fastai.collab import TabularCollab

    original = datasets.DatasetWithTarget, datasets.DatasetWithTargetSpecifyTarget
    original_collab = TabularCollab._dl_type
    for name in ("pimms-dae", "pimms-cf"):
        with pytest.raises(RuntimeError, match="deliberate"):
            with cached_loaders(True, points=100, method=name) as detail:
                assert detail["applied"]
                raise RuntimeError("deliberate")
        assert (datasets.DatasetWithTarget, datasets.DatasetWithTargetSpecifyTarget) == original
        assert TabularCollab._dl_type is original_collab


@pytest.mark.parametrize("method", ["pimms-dae", "pimms-vae", "pimms-cf"])
@pytest.mark.parametrize("seed", [7, 31])
def test_native_fitting_predictions_and_updates_match(method, seed, frame, tmp_path, monkeypatch):
    # Official LR finder, scheduler and callbacks run in both cases. Capture all
    # final parameters as well as predictions; equal update counts alone cannot
    # detect a changed loader shuffle or dropout stream.
    from fastai.learner import Learner

    states, rng_states = [], []
    original = Learner.fit_one_cycle
    def fit(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        states.append({k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()})
        rng_states.append((torch.get_rng_state().clone(), random.getstate(), np.random.get_state()))
        return result
    monkeypatch.setattr(Learner, "fit_one_cycle", fit)
    torch.set_num_threads(1)
    values, hidden = frame
    predictions, details = [], []
    for engine in ("native", "optimized"):
        folder = tmp_path / engine
        folder.mkdir()
        result, detail = run_pimms(method, values.to_numpy(), hidden,
            {"baseline_engine":engine, "max_epochs":2, "batch_size":4,
             "hidden_dims":[8], "latent_dim":4}, device="cpu", seed=seed, folder=folder)
        predictions.append(result)
        details.append(detail)
    assert details[1]["pimms_acceleration"]["applied"]
    assert details[0]["training_steps"] == details[1]["training_steps"]
    assert details[0]["training_steps"] > 0
    assert details[0]["epochs_trained"] == details[1]["epochs_trained"]
    assert states[0].keys() == states[1].keys()
    torch.testing.assert_close(rng_states[0][0], rng_states[1][0], rtol=0, atol=0)
    assert rng_states[0][1] == rng_states[1][1]
    np.testing.assert_array_equal(rng_states[0][2][1], rng_states[1][2][1])
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)
    np.testing.assert_array_equal(predictions[0], predictions[1])
