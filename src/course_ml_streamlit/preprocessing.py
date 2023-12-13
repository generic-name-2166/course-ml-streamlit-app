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

    __slots__ = (
        "numeric_columns",
        "category_columns",
        "resulting_columns",
        "robust_sc",
    )

    def __init__(self):
        columns = self._load_columns()
        self.numeric_columns = columns.get("numeric_columns", [])
        self.category_columns = columns.get("category_columns", [])
        self.resulting_columns = columns.get("resulting_columns", [])
        self.robust_sc = joblib.load("serialized/robust_sc.save")

    @staticmethod
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

        result_df = data_line.drop(columns=self.category_columns)

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

        return _reindex_data(df=result_df, new_index=self.resulting_columns)


def _parse_infrequent_oh_encoded_column(column: str) -> str:
    """
    Changes column name to signify that it's infrequent
    Requires insurance that the input column doesn't have underscores `_`
    """
    return f"{column[:column.rfind("_")]}_infrequent_sklearn"


def _reindex_data(df: pd.DataFrame, new_index: list[str]) -> pd.DataFrame:
    """
    If a column isn't present in the model it gets treated as infrequent.
    Returns DataFrame with columns that fit the model
    """
    resulting_columns = set(new_index)
    columns_not_in_model = set(df.columns) - resulting_columns

    for column_not_in_model in columns_not_in_model:
        if column_not_in_model in resulting_columns:
            continue

        df.drop(columns=column_not_in_model, inplace=True)
        column_infrequent = _parse_infrequent_oh_encoded_column(column_not_in_model)
        # This should theoretically never raise a KeyError
        # because the input columns are predetermined
        df.at[df.index[0], column_infrequent] = 1

    return df.reindex(columns=new_index, fill_value=0, copy=True)


def _config_factory(_instance=PreprocessingConfig()) -> PreprocessingConfig:
    """Theoretically always returns the same instance"""
    return _instance


def preprocess(data_line: pd.DataFrame) -> pd.DataFrame:
    preprocessing_config = _config_factory()
    scaled = preprocessing_config.scale_numeric(data_line=data_line)
    encoded = preprocessing_config.scale_categorical(data_line=scaled)
    return encoded
