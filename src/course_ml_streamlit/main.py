import streamlit as st
import pandas as pd
from course_ml_streamlit.preprocessing import preprocess
from course_ml_streamlit._data import INPUT_DF


def submit(df: pd.DataFrame):
    processed_df = process_input(data_line=df)

    st.session_state.prediction = predict_line(predict_line(data_line=processed_df))


@st.cache_data
def predict_line(data_line: pd.DataFrame) -> bool:
    return False


@st.cache_data
def process_input(data_line: pd.DataFrame) -> pd.DataFrame:
    # TODO: ensure on input that column has no underscores
    return preprocess(data_line=data_line)


def main() -> None:
    st.markdown(
        """
    # Прогноз времени доставки
    Данные для прогноза
    """
    )

    # Transposing DataFrame so that it is easier to edit it
    new_df = st.data_editor(data=INPUT_DF.T, width=1000)

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
            Прогноз для введённых данных - {prediction}.\n
            Поставка займёт **{"больше" if prediction else "меньше"}** чем 3 дня.
            """
        )


if __name__ == "__main__":
    main()
