import course_ml_streamlit.model as mod


# def test_loading():
    # LightGBM segfaults from loaded model
    # _ = mod.load_model()


def test_loading_dt():
    _ = mod.load_model_dt()
