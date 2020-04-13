import pandas as pd
from nltk import FreqDist
from nltk.tokenize import word_tokenize

# Read datasets
sugerencias = pd.read_csv("datasets/sugerencias.csv", parse_dates=["Fecha"]).drop_duplicates()

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

# Word's frequency - missing_info
text1 = "\n".join(sugerencias["missing_info"].dropna().values)
words1 = word_tokenize(text1)
fdist1 = FreqDist(words1)
print(fdist1.most_common(50))
fdist1.plot(50)

# Word's frequency - improvements
text2 = "\n".join(sugerencias["improvements"].dropna().values)
words2 = word_tokenize(text2)
fdist2 = FreqDist(words2)
print(fdist2.most_common(50))
fdist2.plot(50)