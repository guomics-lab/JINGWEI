"""Scientific invariants of the corrected within-protein MNAR rule."""
import numpy as np
import pytest

from jingwei_benchmark.protocol import (JINGWEI_MECHANISM, missing_probabilities,
                                      protocol_masks, uses_mask_rate)


def test_jingwei_mnar_is_strictly_abundance_dependent_within_each_protein():
    values = np.column_stack([np.linspace(10, 20, 100), np.linspace(30, 31, 100)])
    observed = np.ones(values.shape, dtype=bool)
    p = missing_probabilities(values, observed, JINGWEI_MECHANISM)
    assert (np.diff(p, axis=0) < 0).all()
    assert np.ptp(p[:, 0]) > .5
    np.testing.assert_allclose(p[:, 0], p[:, 1], atol=1e-13)
    # Changing units and offsets must not replay the historical all-p=0.9 bug.
    shifted = missing_probabilities(values * [3, .2] + [1000, -100], observed, JINGWEI_MECHANISM)
    np.testing.assert_allclose(p, shifted, atol=1e-12)


@pytest.mark.parametrize('rate', [1e-30, .01, .2, .6, .95, 1 - 1e-12])
def test_requested_rate_is_calibrated_without_destroying_mnar(rate):
    values = np.random.default_rng(16).normal(20, 2, (1000, 10))
    observed = np.ones(values.shape, dtype=bool)
    masks, meta = protocol_masks(values, observed, mechanism=JINGWEI_MECHANISM,
                                evaluation_rate=rate, training_rate=0, seed=9,
                                preserve_marginals=False)
    stage = meta['evaluation']
    assert stage['probability_mean'] == pytest.approx(rate, abs=1e-12)
    assert abs(stage['actual_rate'] - rate) < .02
    assert stage['probability_min'] < stage['probability_max']
    assert stage['rule_version'] == 2 and stage['value_scale'] == 'working'
    assert not masks.training_hidden.any()
    assert uses_mask_rate(JINGWEI_MECHANISM, uncalibrated=True)


def test_low_abundance_values_are_more_frequently_hidden_in_a_single_common_mask():
    values = np.tile(np.linspace(10, 20, 1000)[:, None], (1, 30))
    masks, _ = protocol_masks(values, np.ones(values.shape, dtype=bool),
                              mechanism=JINGWEI_MECHANISM, evaluation_rate=.2,
                              training_rate=0, seed=4)
    assert masks.evaluation_hidden[:200].mean() > 3 * masks.evaluation_hidden[-200:].mean()


def test_missing_intensities_do_not_affect_probabilities_and_constant_columns_work():
    values = np.random.default_rng(5).normal(20, 1, (100, 4))
    observed = np.ones(values.shape, dtype=bool)
    observed[::3, 0] = False
    observed[:, 2] = False
    values[:, 3] = 8
    p = missing_probabilities(values, observed, JINGWEI_MECHANISM)
    values[~observed] = np.nan
    np.testing.assert_array_equal(p, missing_probabilities(values, observed, JINGWEI_MECHANISM))
    assert not p[~observed].any()
    np.testing.assert_allclose(p[:, 3], .5)


def test_v2_mask_is_based_on_working_scale_and_never_hides_native_missing_values():
    working = np.random.default_rng(2).normal(20, 1, (50, 7))
    observed = np.ones(working.shape, dtype=bool)
    observed[:5, 0] = False
    a, _ = protocol_masks(2. ** working, observed, working_values=working,
                          mechanism=JINGWEI_MECHANISM, training_rate=0, seed=4)
    b, _ = protocol_masks(working, observed, working_values=working,
                          mechanism=JINGWEI_MECHANISM, training_rate=0, seed=4)
    np.testing.assert_array_equal(a.evaluation_hidden, b.evaluation_hidden)
    assert not (a.evaluation_hidden & ~observed).any()


def test_gui_v2_ignores_irrelevant_custom_quantile_settings(tmp_path):
    from jingwei_benchmark.gui_jobs import make_argv
    config = {'command':'benchmark','methods':['mean'],'device':'cpu',
              'mechanism':JINGWEI_MECHANISM,'mask_rate':.2,
              'low_quantile':.9,'high_quantile':.1}
    argv = make_argv(config,tmp_path/'input.csv',tmp_path/'output')
    assert JINGWEI_MECHANISM in argv


def test_gui_v2_rate_is_editable_and_old_form_migrates_without_rewriting_old_tasks(tmp_path, monkeypatch):
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication
    from jingwei_benchmark.gui import BenchmarkWindow, set_value, value
    from jingwei_benchmark.gui_jobs import write_json, read_json
    from jingwei_benchmark.protocol import ORIGINAL_MECHANISM
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(BenchmarkWindow, 'refresh_environment', lambda self: None)
    write_json(tmp_path / 'session.json', {'schema':1, 'fields':{
        'mechanism':ORIGINAL_MECHANISM, 'mask_rate':.35}})
    old_task = tmp_path / 'old-task.json'
    old_task.write_text('{"mechanism":"mnar-jingwei-original"}')
    win = BenchmarkWindow(tmp_path, load_default=False)
    try:
        assert value(win.fields['mechanism']) == JINGWEI_MECHANISM
        for task in ['benchmark','impute']:
            win.fields['command'].setCurrentIndex(win.fields['command'].findData(task))
            assert win.fields['mask_rate'].isEnabled() and win.mask_rate_stack.currentIndex() == 0
            assert value(win.fields['mask_rate']) == pytest.approx(.35)
        win.set_language('en')
        assert win.fields['mechanism'].currentText() == 'MNAR · JINGWEI'
        assert win.fields['mask_rate'].isEnabled()
    finally:
        win.close()
    assert read_json(tmp_path / 'session.json')['mnar_rule_version'] == 2
    assert read_json(old_task)['mechanism'] == ORIGINAL_MECHANISM
