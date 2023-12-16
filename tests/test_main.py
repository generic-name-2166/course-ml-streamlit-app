import pytest
import course_ml_streamlit.main as app
from course_ml_streamlit._data import INPUT_DF, UNDERSCORES
import pandas as pd


def add_underscore(underscore_value: str) -> pd.DataFrame:
    df = INPUT_DF.copy()
    for col in df.columns:
        if isinstance(df.at[0, col], str):
            df.at[0, col] = underscore_value
    return df


PRE_DF = pd.read_csv(
    filepath_or_buffer="serialized/test_sample_pre.csv", header=0, index_col=0
)
UNDERSCORE_DFS = [add_underscore(value) for value in UNDERSCORES]


def test_predicting():
    assert isinstance(PRE_DF, pd.DataFrame)
    prediction = app.predict_line(data_line=PRE_DF)
    assert isinstance(prediction, int), type(prediction)


@pytest.mark.parametrize("data_line", [INPUT_DF] + UNDERSCORE_DFS)
def test_processing(data_line):
    processed = app.process_input(data_line=data_line)
    assert isinstance(processed, pd.DataFrame)


def test_submit():
    app.submit(df=INPUT_DF)
    # assert isinstance(st.session_state.prediction, bool)
    # can't access session_state from this file
    # so this is mostly redundant
