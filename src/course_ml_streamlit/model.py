import joblib
import streamlit as st
# import lightgbm as lgb

"""
# LightGBM segafaults from loaded model
@st.cache_resource
def load_model():
    return lgb.Booster(model_file="serialized/gbm_classifier_t.txt")
"""


@st.cache_resource
def load_model_dt():
    return joblib.load("serialized/dt_classifier.save")
