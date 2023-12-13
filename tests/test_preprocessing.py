import pytest
import course_ml_streamlit.preprocessing as c_pre
import pandas as pd


SAMPLE_DATA = [
    [
        86420,
        84480,
        'ООО "Торговая компания "ЕвразХолдинг"',
        5003,
        5757676,
        5191,
        0,
        32411,
        67.46,
        1,
        45536,
        "Экспорт",
        86420,
        643,
        417,
    ],
    [
        65540,
        65700,
        "(Служба пути) Куйбышевская дирекция инфраструктуры-структурное подразд",
        9913,
        69079506,
        9960,
        69079506,
        32102,
        201.562,
        62,
        30442,
        "Внутр. пер",
        0,
        0,
        0,
    ],
    [
        81760,
        20020,
        'Публичное акционерное общество "Магнитогорский металлургический комбин',
        5010,
        186424,
        5441,
        897013,
        32418,
        67.52,
        1,
        151245,
        "Внутр. пер",
        0,
        0,
        0,
    ],
    [
        59220,
        -1,
        "test",
        3437,
        -1,
        -1,
        -1,
        -1,
        67.52,
        1,
        151245,
        "Внутр. пер",
        0,
        0,
        0,
    ],
]
SAMPLE_COLUMNS = [
    "Код станции отправления",
    "Код станции назначения",
    "Клиент",
    "Код отправителя груза",
    "ОКПО отправителя",
    "Код получателя груза",
    "ОКПО получателя",
    "Код груза",
    "Вес груза, тонн",
    "Количество вагонов",
    "Тонно-километры",
    "Характер перевозок",
    "Код станции отправления загран",
    "Код страны отправления",
    "Код страны назначения",
]


def test_instances():
    a = c_pre._config_factory()
    b = c_pre._config_factory()
    assert a is b


def test_infrequent():
    """ Test that columns are claimed as infrequent correctly"""
    sample_controlled = pd.DataFrame(
        data=[SAMPLE_DATA[-1]], columns=SAMPLE_COLUMNS
    )
    preprocessed = c_pre.preprocess(data_line=sample_controlled)
    assert preprocessed.at[preprocessed.index[0],
                           "Код_станции_отправления_59220"] == 1
    assert preprocessed.at[preprocessed.index[0],
                           "Клиент_infrequent_sklearn"] == 1
    assert preprocessed.at[preprocessed.index[0],
                           "Код_отправителя_груза_3437"] == 1


@pytest.mark.parametrize(
    "data_line",
    [
        pd.DataFrame(
            data=[SAMPLE_DATA[i]],
            columns=SAMPLE_COLUMNS,
        )
        for i, _ in enumerate(SAMPLE_DATA)
    ],
)
def test_preprocessing(data_line: pd.DataFrame):
    """ Test that the resulting data is compatible with model """
    assert isinstance(data_line, pd.DataFrame)
    assert data_line.shape == (1, 15)
    result_df = c_pre.preprocess(data_line=data_line)
    assert len(result_df.columns) == 82
