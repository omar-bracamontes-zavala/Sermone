import pandas as pd

# Read datasets
sugerencias = pd.read_csv("datasets/sugerencias.csv", parse_dates=["Fecha"]).drop_duplicates()
# homoclaves = pd.read_csv("datasets/homoclaves.csv") # We don't need this dataset

# Analyze relation between column E and F
x = sugerencias[sugerencias["1.2 ¿Consideras que falta información?"] == "No"]["1.3 ¿Qué información crees que falta?"]
print(x.dropna())

# Drop unnecessary columns
sugerencias.drop(["Encuesta", "Ficha", "1.2 ¿Consideras que falta información?"], axis=1, inplace=True)

# Rename columns
sugerencias.columns = ["institution", "is_info_useful", "missing_info", "improvements", "date"]

# Transform "homoclave" field into "institution" field so we can have the institution name
sugerencias["institution"] = sugerencias["institution"].apply(lambda string: string.split('-')[0])
# Map is_info_useful to boolean values
sugerencias["is_info_useful"] = sugerencias["is_info_useful"].map({"Sí": 1, "No": 0})
