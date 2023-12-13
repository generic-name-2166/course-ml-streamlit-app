import pandas as pd
import course_ml_streamlit.preprocessing as prepro
import numpy as np


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


def round_floats(value):
    if isinstance(value, float) or isinstance(value, np.float_):
        return round(value, 7)
    return value


df0 = pd.DataFrame(data=[SAMPLE_DATA[0]], columns=SAMPLE_COLUMNS)
df1 = pd.DataFrame(data=[SAMPLE_DATA[1]], columns=SAMPLE_COLUMNS)
df2 = pd.DataFrame(data=[SAMPLE_DATA[2]], columns=SAMPLE_COLUMNS)
df3 = pd.DataFrame(data=[SAMPLE_DATA[3]], columns=SAMPLE_COLUMNS)
df = pd.DataFrame(data=SAMPLE_DATA, columns=SAMPLE_COLUMNS)
pre0 = prepro.preprocess(df0)
pre1 = prepro.preprocess(df1)
pre2 = prepro.preprocess(df2)
pre3 = prepro.preprocess(df3)
print(df3)
print(pre3)
pre = pd.concat([pre0, pre1, pre2, pre3])

pre.to_csv(
    path_or_buf="serialized/test_sample_pre.csv",
    columns=pre.columns,
    header=True,
    index=True,
    mode="w",
    compression=None,
)
df.to_csv(
    path_or_buf="serialized/test_sample.csv",
    columns=df.columns,
    header=True,
    index=True,
    mode="w",
    compression=None,
)
deser = pd.read_csv(
    filepath_or_buffer="serialized/test_sample_pre.csv", header=0, index_col=0
)
# print(deser.map(round_floats) == pre.map(round_floats))
print(not pre.isnull().to_numpy().any())
print(deser.iat[0, 1], pre.iat[0, 1])
