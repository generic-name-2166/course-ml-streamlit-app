import joblib


def load_model():
    return joblib.load("serialized/gbm_classifier_t.txt")
