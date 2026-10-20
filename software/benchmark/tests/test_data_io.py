import numpy as np
import pandas as pd
import pytest
from jingwei_benchmark.data_io import load_matrix, write_predictions


def test_explicitly_excluded_features_remain_missing_and_do_not_hide_other_failures(tmp_path):
    source = tmp_path / "scope.csv"
    pd.DataFrame([[2., 3.], [4., 5.]], columns=["P0", "P1"]).to_csv(source, index=False)
    data = load_matrix(source, transform="none")
    values = np.array([[2., np.nan], [4., np.nan]])
    write_predictions(data, values, tmp_path, method="mean", input_scale="raw",
                      excluded_features=np.array([False, True]))
    exported = pd.read_csv(tmp_path / "imputed.csv", index_col=0).to_numpy()
    assert np.isnan(exported[:, 1]).all()
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite predictions"):
        write_predictions(data, values, tmp_path, method="mean", input_scale="raw",
                          excluded_features=np.array([False, True]))


def test_csv_without_identifier_column_keeps_first_feature(tmp_path):
    source = tmp_path / "input.csv"
    pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=["P1", "P2"]).to_csv(
        source, index=False
    )
    result = load_matrix(source, index_column="none", transform="none")
    assert list(result.frame.columns) == ["P1", "P2"]
    assert result.frame.shape == (2, 2)


def test_transposed_csv_and_negative_log_values(tmp_path):
    source = tmp_path / "input.csv"
    pd.DataFrame(
        [[-2.0, 3.0], [0.0, 4.0]], index=["P1", "P2"], columns=["S1", "S2"]
    ).to_csv(source)
    result = load_matrix(source, orientation="features-by-samples", scale="log2")
    assert result.observed.all()
    assert result.frame.loc["S1", "P1"] == -2
    with pytest.raises(ValueError, match="already"):
        load_matrix(source, scale="log2", transform="log2")


def test_zero_policy_is_explicit(tmp_path):
    source = tmp_path / "input.csv"
    pd.DataFrame(
        [[0.0, 2.0], [3.0, 4.0]], index=["S1", "S2"], columns=["P1", "P2"]
    ).to_csv(source)
    assert load_matrix(source, transform="none").observed.all()
    assert not load_matrix(source, missing_values="zero", transform="none").observed[
        0, 0
    ]


@pytest.mark.parametrize(
    "contents,shape,has_header,has_id",
    [
        ("ID,P1,P2\nS1,1,2\nS2,3,4\n", (2, 2), True, True),
        ("P1,P2\n1,2\n3,4\n", (2, 2), True, False),
        ("1,2\n3,4\n", (2, 2), False, False),
        ("S1,1,2\nS2,3,4\n", (2, 2), False, True),
        ("Sample,1,2\nS1,3,4\nS2,5,6\n", (2, 2), True, True),
        ("P1,P2\nNA,2\n3,4\n", (2, 2), True, False),
    ],
)
def test_auto_csv_layout_preserves_all_expression_cells(
    tmp_path, contents, shape, has_header, has_id
):
    source = tmp_path / "matrix.csv"
    source.write_text(contents)
    result = load_matrix(source, transform="none")
    assert result.frame.shape == shape
    assert result.csv_detection["has_header"] == has_header
    assert result.csv_detection["has_id_column"] == has_id


def test_numeric_identifiers_require_explicit_override(tmp_path):
    source = tmp_path / "matrix.csv"
    source.write_text("ID,P1,P2\n101,1,2\n102,3,4\n")
    automatic = load_matrix(source, transform="none")
    explicit = load_matrix(source, index_column="first", transform="none")
    assert automatic.frame.shape == (2, 3)
    assert explicit.frame.shape == (2, 2)
    assert list(explicit.frame.index) == ["101", "102"]


def test_numeric_feature_headers_can_be_overridden(tmp_path):
    source = tmp_path / "matrix.csv"
    source.write_text("100,200\n1,2\n3,4\n")
    assert load_matrix(source, header="yes", transform="none").frame.shape == (2, 2)


def test_h5ad_layer_and_annotations_survive_output(tmp_path):
    ad = pytest.importorskip("anndata")
    source = tmp_path / "input.h5ad"
    original = ad.AnnData(np.array([[1.0, 2.0], [3.0, 4.0]]))
    original.obs["group"] = ["A", "B"]
    original.var["symbol"] = ["X", "Y"]
    original.layers["expression"] = np.array([[2.0, 4.0], [6.0, 8.0]])
    original.write_h5ad(source)
    data = load_matrix(source, layer="expression", transform="none")
    write_predictions(
        data, data.frame.to_numpy(), tmp_path, method="mean", input_scale="raw"
    )
    result = ad.read_h5ad(tmp_path / "imputed.h5ad")
    np.testing.assert_array_equal(result.X, original.X)
    np.testing.assert_array_equal(
        result.layers["jingwei_imputed"], original.layers["expression"]
    )
    assert result.obs["group"].tolist() == ["A", "B"]
    assert result.var["symbol"].tolist() == ["X", "Y"]
