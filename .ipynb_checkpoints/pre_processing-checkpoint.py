from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd


class AttributesTransformator(BaseEstimator, TransformerMixin):
    def __init__(self, days_transformer = True, boolean_mapper = True, tilde_remover = True):
        self.days_transformer = days_transformer
        self.boolean_mapper = boolean_mapper
        self.tilde_remover = tilde_remover

    @staticmethod
    def date_to_days(date_series):
        min_date = date_series.min()
        return date_series.apply(lambda x: (x - min_date).days)

    @staticmethod
    def remove_tildes(string):
        if string == "nan":
            return ""
        string = string.lower()
        vocales = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u"}
        for vocal in vocales.keys():
            string = string.replace(vocal, vocales[vocal])
        return string

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        if self.days_transformer:
            df["date"] = self.date_to_days(df["date"])
        if self.boolean_mapper:
            df["is_info_useful"] = df["is_info_useful"].map({"Sí": 1, "No": 0})
        if self.tilde_remover:
            df["missing_info"] = df["missing_info"].apply(lambda x: self.remove_tildes(x))
            df["improvements"] = df["improvements"].apply(lambda x: self.remove_tildes(x))
        return df


def load_data(file_name="datasets/sugerencias.csv"):
    useful_cols = ["Fecha", "Homoclave", "1.1 ¿Te parece útil esta información?",
                   "1.3 ¿Qué información crees que falta?",
                   "1.4 ¿Qué podemos mejorar?"]
    df = pd.read_csv(file_name, parse_dates=["Fecha"],
                     usecols=useful_cols, encoding='utf-8').drop_duplicates()
    df.columns = ["institution", "is_info_useful", "missing_info", "improvements", "date"]
    df["missing_info"] = df["missing_info"].astype(str)
    df["improvements"] = df["improvements"].astype(str)
    df.sort_values('date', ascending=True, inplace=True)
    na_filter = df["missing_info"].isna() & df["improvements"].isna()
    return df[~na_filter]


if __name__ == "__main__":
    df = load_data()

    transf_attribs = ["is_info_useful", "missing_info", "improvements", "date"]
    cat_attribs = ["institution"]
    preparation_pipeline = ColumnTransformer([
        ("transform", AttributesTransformator(), transf_attribs),
        ("encode", OrdinalEncoder(), cat_attribs),
    ])
    df_prepared = preparation_pipeline.fit_transform(df)
    print(df_prepared)
