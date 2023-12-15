import pytest
import course_ml_streamlit.preprocessing as c_pre
from course_ml_streamlit._data import SAMPLE_DATA, SAMPLE_COLUMNS
import pandas as pd


def test_instances():
    a = c_pre._config_factory()
    b = c_pre._config_factory()
    assert a is b


def test_infrequent():
    """Test that columns are claimed as infrequent correctly"""
    sample_controlled = pd.DataFrame(data=[SAMPLE_DATA[-1]], columns=SAMPLE_COLUMNS)
    preprocessed = c_pre.preprocess(data_line=sample_controlled)
    assert preprocessed.at[preprocessed.index[0], "Код_станции_отправления_59220"] == 1
    assert preprocessed.at[preprocessed.index[0], "Клиент_infrequent_sklearn"] == 1
    assert preprocessed.at[preprocessed.index[0], "Код_отправителя_груза_3437"] == 1


def test_shape():
    test_df = pd.DataFrame(data=[["Test"]] * 15, columns=["Test"])
    with pytest.raises(ValueError):
        _ = c_pre.preprocess(test_df)


@pytest.mark.parametrize(
    "data_line",
    [
        pd.DataFrame(
            data=[SAMPLE_DATA[i]],
            columns=SAMPLE_COLUMNS,
        )
        for i, _ in enumerate(SAMPLE_DATA)
    ],
)
def test_preprocessing(data_line: pd.DataFrame):
    """Test that the resulting data is compatible with model"""
    assert isinstance(data_line, pd.DataFrame)
    assert data_line.shape == (1, 15)
    result_df = c_pre.preprocess(data_line=data_line)
    assert len(result_df.columns) == 82
    assert result_df.at[0, "Вес груза, тонн"] != 0
    assert not result_df.isnull().to_numpy().any()
