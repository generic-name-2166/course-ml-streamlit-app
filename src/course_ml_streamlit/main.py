import streamlit as st
import pandas as pd
from course_ml_streamlit.preprocessing import preprocess


def process_input(data_line: pd.DataFrame) -> pd.DataFrame:
    # TODO: ensure on input that column has no underscores
    return preprocess(data_line=data_line)


def main() -> None:
    df = pd.DataFrame({"first column": [1, 2, 3, 4], "second column": [10, 20, 30, 40]})

    st.write("""gaming""")

    st.write(df)


if __name__ == "__main__":
    main()
