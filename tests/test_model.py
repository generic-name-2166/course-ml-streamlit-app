import pytest
import course_ml_streamlit.model as mod


@pytest.mark.skip
def test_loading():
    _ = mod.load_model()
