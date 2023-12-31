import streamlit as st
import pandas as pd
from course_ml_streamlit.preprocessing import preprocess
from course_ml_streamlit.model import load_model_dt
from course_ml_streamlit._data import INPUT_DF


def submit(df: pd.DataFrame):
    """
    Stores model prediction in streamlit.session_state.prediction
    """
    processed_df = process_input(data_line=df)

    st.session_state.prediction = predict_line(data_line=processed_df)


@st.cache_data
def predict_line(data_line: pd.DataFrame) -> int:
    model = load_model_dt()

    pred = model.predict(X=data_line.to_numpy())
    return int(pred[0])


@st.cache_data
def process_input(data_line: pd.DataFrame) -> pd.DataFrame:
    """
    Check data for things such as extra underscores 
    which will mess with the processing
    """
    idx = data_line.index[0]
    for column in data_line.columns:
        value = data_line.at[idx, column]
        if isinstance(value, str) and "_" in value:
            data_line.at[idx, column] = value.replace("_", " ")

    return preprocess(data_line=data_line)


def main() -> None:
    st.markdown(
        """
    # Прогноз времени доставки
    Данные для прогноза
    """
    )

    # Transposing DataFrame so that it is easier to edit it
    new_df = st.data_editor(data=INPUT_DF.T, width=1000, height=300)

    # DataFrame is transposed back for model
    st.button(
        label="Predict",
        on_click=submit,
        args=(new_df.T,),
        type="primary",
        use_container_width=True,
    )

    if "prediction" in st.session_state:
        prediction = st.session_state.prediction
        st.markdown(
            body=f"""
            #### Прогноз для введённых данных - {bool(prediction)}.\n
            #### Поставка займёт {"более" if prediction else "менее"} 3 дней.
            """
        )


if __name__ == "__main__":
    main()
