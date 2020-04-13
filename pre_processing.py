import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


def load_data(file_name="datasets/sugerencias.csv"):
    useful_cols = ["Fecha", "Homoclave", "1.1 ¿Te parece útil esta información?",
                   "1.3 ¿Qué información crees que falta?",
                   "1.4 ¿Qué podemos mejorar?"]
    df = pd.read_csv(file_name, parse_dates=["Fecha"],
                     usecols=useful_cols, encoding='utf-8').drop_duplicates()
    df.columns = ["institution", "is_info_useful", "missing_info", "improvements", "date"]
    df.sort_values('date', ascending=True, inplace=True)
    return df


def date_to_days(date_series):
    min_date = date_series.min()
    return date_series.apply(lambda x: (x - min_date).days)


def remove_tildes(string):
    if string == "nan":
        return ""
    string = string.lower()
    vocales={"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u"}
    for vocal in vocales.keys():
        string = string.replace(vocal, vocales[vocal])
    return string


df = load_data()
na_filter = df["missing_info"].isna() & df["improvements"].isna()
df = df[~na_filter]
df["date"] = date_to_days(df["date"])
df["is_info_useful"] = df["is_info_useful"].map({"Sí": 1, "No": 0})
df["missing_info"] = df["missing_info"].astype(str)
df["improvements"] = df["improvements"].astype(str)
df.loc[:, "missing_info"] = df["missing_info"].apply(lambda x: remove_tildes(x))
df.loc[:, "improvements"] = df["improvements"].apply(lambda x: remove_tildes(x))

from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()

df['institution_cat'] = ordinal_encoder.fit_transform(df[['institution']])
df.drop(["institution"], axis=1, inplace=True)