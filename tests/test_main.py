import pytest
import course_ml_streamlit.main as app
import pandas as pd


@pytest.mark.skip
def test_processing():
    _ = app.process_input(data_line=pd.DataFrame([]))


@pytest.mark.skip
def test_predicting():
    _ = app.predict_line(data_line=pd.DataFrame([]))
