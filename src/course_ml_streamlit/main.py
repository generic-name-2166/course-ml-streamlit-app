import streamlit as st
import pandas as pd
from course_ml_streamlit.preprocessing import preprocess


def predict_line(data_line: pd.DataFrame) -> bool:
    pass


def process_input(data_line: pd.DataFrame) -> pd.DataFrame:
    # TODO: ensure on input that column has no underscores
    # TODO: ensure here preprocess doesn't throw
    try:
        # A crutch for now
        return preprocess(data_line=data_line)
    except ValueError:
        return data_line


def main() -> None:
    df = pd.DataFrame({"first column": [1, 2, 3, 4], "second column": [10, 20, 30, 40]})

    st.write("""gaming""")

    st.write(df)

    new_df = st.data_editor(data=df.iloc[0])

    processed_df = process_input(data_line=new_df)

    prediction = predict_line(predict_line(data_line=processed_df))

    st.markdown(body=f"""
    The prediction for the input data is {prediction}.\n
    The delivery will take *{"longer" if prediction else "shorter"}* than 3 days.
    """)

if __name__ == "__main__":
    main()
