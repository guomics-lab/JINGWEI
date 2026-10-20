"""User-facing input contracts, including paths and labels from spreadsheet exports."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jingwei_benchmark.cli import main
from jingwei_benchmark.data_io import load_matrix


@pytest.mark.parametrize('layout', ['bom', 'header-only', 'numeric-only', 'transposed', 'numeric-ids'])
def test_csv_user_layouts_keep_every_value_and_identifier(tmp_path, capsys, layout):
    source = tmp_path / '用户 数据.csv'
    expected = np.arange(1., 13.).reshape(3, 4)
    frame = pd.DataFrame(expected, index=['001', '010', '100'], columns=['P1', 'P2', 'P3', 'P4'])
    options = ['--transform', 'none']
    if layout == 'bom':
        frame.index = ['样本 A', '样本 B', '样本 C']
        frame.to_csv(source, encoding='utf-8-sig')
    elif layout == 'header-only':
        frame.to_csv(source, index=False)
    elif layout == 'numeric-only':
        frame.to_csv(source, header=False, index=False)
    elif layout == 'transposed':
        frame.index = ['S1', 'S2', 'S3']
        frame.T.to_csv(source)
        options += ['--orientation', 'features-by-samples']
    else:
        frame.to_csv(source)
        options += ['--index-column', 'first']
    assert main(['inspect', '--input', str(source), *options]) == 0
    inspection = json.loads(capsys.readouterr().out)
    assert (inspection['samples'], inspection['features']) == expected.shape
    keywords = {'transform': 'none'}
    if layout == 'transposed':
        keywords['orientation'] = 'features-by-samples'
    if layout == 'numeric-ids':
        keywords['index_column'] = 'first'
    data = load_matrix(source, **keywords)
    np.testing.assert_array_equal(data.frame, expected)
    if layout in ('numeric-ids', 'bom', 'transposed'):
        assert data.frame.index.tolist() == frame.index.tolist()


def test_leading_blank_lines_in_spreadsheet_csv_do_not_hide_header(tmp_path):
    source = tmp_path / 'blank-lines.csv'
    source.write_text('\ufeff\n  \nID,P1,P2\nA,1,2\n\nB,3,4\n', encoding='utf-8')
    data = load_matrix(source, transform='none')
    assert data.frame.index.tolist() == ['A', 'B']
    np.testing.assert_array_equal(data.frame, [[1, 2], [3, 4]])


@pytest.mark.parametrize('contents', ['', 'ID,P1,P2\nA,1\nB,2,3\n',
    'ID,P1,P2\nA,1,2\nA,3,4\n', 'ID,P1,P1\nA,1,2\nB,3,4\n',
    'ID,P1,P2\nA,inf,2\nB,3,4\n', 'ID,P1,P2\nA,bad,2\nB,3,4\n',
    'ID,P1,P2\nA,,\nB,,\n'])
def test_bad_csv_reports_a_controlled_error_without_traceback(tmp_path, capsys, contents):
    source = tmp_path / 'bad.csv'
    source.write_text(contents)
    assert main(['inspect', '--input', str(source)]) == 1
    captured = capsys.readouterr()
    assert 'jingwei:' in captured.err
    assert 'Traceback' not in captured.err


def test_csv_dense_memory_budget_is_enforced_before_pandas_allocates(tmp_path, monkeypatch):
    source = tmp_path / 'large.csv'
    pd.DataFrame(np.ones((400, 400))).to_csv(source, index=False)
    monkeypatch.setattr(pd, 'read_csv', lambda *a, **kw: pytest.fail('oversize matrix reached pandas'))
    with pytest.raises(ValueError, match='max-dense-mb'):
        load_matrix(source, max_dense_mb=1)


@pytest.mark.parametrize('limit', [0, -1])
def test_dense_memory_budget_must_be_positive(tmp_path, limit):
    source = tmp_path / 'input.csv'
    source.write_text('P1,P2\n1,2\n3,4\n')
    with pytest.raises(ValueError, match='max-dense-mb'):
        load_matrix(source, max_dense_mb=limit)


@pytest.mark.parametrize('config', [{'mean': []}, {'mean': 7}, {'mean': None}, []])
def test_malformed_method_config_is_a_controlled_cli_error(tmp_path, capsys, config):
    source = tmp_path / 'input.csv'
    source.write_text('P1,P2,P3\n1,2,3\n4,5,6\n7,8,9\n')
    settings = tmp_path / 'methods.json'
    settings.write_text(json.dumps(config))
    output = tmp_path / 'results'
    assert main(['benchmark', '--input', str(source), '--methods', 'mean', '--method-config', str(settings),
                 '--repeats', '1', '--device', 'cpu', '--output', str(output)]) == 1
    assert 'Traceback' not in capsys.readouterr().err
    assert not (output / 'repeat_000').exists()


@pytest.mark.parametrize('timeout', ['nan', 'inf', '-inf'])
def test_nonfinite_timeout_is_rejected_before_launch(tmp_path, capsys, monkeypatch, timeout):
    import jingwei_benchmark.benchmark as backend
    monkeypatch.setattr(backend, 'run_experiment', lambda args: pytest.fail('nonfinite timeout reached experiment'))
    assert main(['benchmark', '--input', str(tmp_path / 'input.csv'), '--output', str(tmp_path / 'out'),
                 '--timeout=' + timeout]) == 1
    assert 'timeout' in capsys.readouterr().err


def test_anndata_without_X_reports_a_controlled_error_and_can_use_a_layer(tmp_path, capsys):
    ad=pytest.importorskip('anndata')
    source=tmp_path/'layer-only.h5ad'
    values=np.arange(1.,13.).reshape(3,4)
    data=ad.AnnData(shape=values.shape)
    data.layers['abundance']=values
    data.write_h5ad(source)
    assert main(['inspect','--input',str(source)])==1
    error=capsys.readouterr().err
    assert 'X' in error and '--layer' in error and 'Traceback' not in error
    assert main(['inspect','--input',str(source),'--layer','abundance'])==0
    result=json.loads(capsys.readouterr().out)
    assert (result['samples'],result['features'])==(3,4)
    np.testing.assert_array_equal(load_matrix(source,layer='abundance',transform='none').frame,values)
