import sklearn
import joblib
import json
import pandas as pd
import numpy as np


class PreprocessingConfig:
    """
    A singleton responsible for preprocessing data for prediction model
    Made into a class for convenience
    """

    __slots__ = ("numeric_columns", "category_columns", "resulting_columns", "robust_sc")

    def __init__(self):
        columns = self._load_columns()
        self.numeric_columns = columns.get("numeric_columns", [])
        self.category_columns = columns.get("category_columns", [])
        self.resulting_columns = columns.get("resulting_columns", [])
        self.robust_sc = joblib.load("serialized/robust_sc.save")

    def _load_columns() -> dict:
        with open("serialized/columns_export.json") as F:
            return json.loads(F.read())

    def scale_numeric(self, data_line: pd.DataFrame) -> pd.DataFrame:
        """
        Scales numeric columns DataFrame and returns it
        Always assumes the data is just 1 line
        """

        data_line[self.numeric_columns] = self.robust_sc.fit_transform(
            data_line[self.numeric_columns]
        )
        return data_line

    def scale_categorical(self, data_line: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes categorical columns with sklearn.preprocessing.OneHotEncoder
        Always assumes the data is just 1 line
        """

        result_df = data_line.drop(self.category_columns)

        for col in self.category_columns:
            oh_encoder = sklearn.preprocessing.OneHotEncoder(
                categories="auto",
                drop="if_binary",
                sparse_output=True,
                dtype=np.int_,
                handle_unknown="error",
                min_frequency=0.02,
                max_categories=None,
                feature_name_combiner="concat",
            )

            encoded_sparse = oh_encoder.fit_transform(
                data_line[col].to_numpy().reshape(-1, 1)
            ).toarray()
            temp_df = pd.DataFrame(
                data=encoded_sparse,
                columns=oh_encoder.get_feature_names_out([col.replace(" ", "_")]),
            )
            result_df = pd.concat([result_df, temp_df], axis=1)

        # TODO: Handle extra columns into infrequent

        return result_df


def _config_factory(_instance=PreprocessingConfig()) -> PreprocessingConfig:
    """Theoretically always returns the same instance"""
    return _instance


def preprocess(data_line: pd.DataFrame) -> pd.DataFrame:
    preprocessing_config = _config_factory()
    scaled = preprocessing_config.scale_numeric(data_line=data_line)
    encoded = preprocessing_config.scale_categorical(data_line=scaled)
    return encoded
