import pytest
import course_ml_streamlit.preprocessing as c_pre
import pandas as pd


def test_instances():
    a = c_pre._config_factory()
    b = c_pre._config_factory()
    assert a is b


@pytest.mark.skip(reason="columns aren't instantiated correctly yet")
@pytest.mark.parametrize("data_line", [
    (pd.DataFrame(data=[], columns=[]))
])
def test_preprocessing(data_line: pd.DataFrame):
    result_df = c_pre.preprocess(data_line=data_line)
    assert len(result_df.columns) == 82
