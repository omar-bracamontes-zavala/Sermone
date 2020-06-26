
from google.cloud import language_v1
from google.cloud.language_v1 import enums
from google.cloud import storage
from time import sleep
import pandas
import sys

pandas.options.mode.chained_assignment = None

def parse_csv_from_gcs(csv_file):
    df = pandas.read_csv(csv_file, encoding = "ISO-8859-1")

    return df

def rename_schema(dataframe):
    dataframe.columns = ['Encuesta', 'Ficha', 'Homoclave', 'isUseful', 'isLackingInfo', 'InformacionFaltante', 'PosiblesMejoras', 'Fecha', 'Empty1', 'Empty2']

def cleanup_missing_info_field(dataframe):
    dataframe.dropna(subset=["PosiblesMejoras"], inplace = True)

def add_organization_column(dataframe):
    dataframe['Organization'] = dataframe['Homoclave'].str.split("-").str[0]

def analyze_sentiment(text_content, client):
    type_ = enums.Document.Type.PLAIN_TEXT
    language = 'es'
    document = {"content": text_content, "type": type_, "language": language}
    encoding_type = enums.EncodingType.UTF8
    response = client.analyze_sentiment(document, encoding_type=encoding_type)

    return response

gcs_path = sys.argv[1]
output_bucket = sys.argv[2]
output_csv_file = sys.argv[3]

organization_dataframe = parse_csv_from_gcs(gcs_path)
rename_schema(organization_dataframe)
cleanup_missing_info_field(organization_dataframe)

add_organization_column(organization_dataframe)

client = language_v1.LanguageServiceClient()

for i in organization_dataframe.index:
    response = analyze_sentiment(organization_dataframe.at[i, 'PosiblesMejoras'], client)
    organization_dataframe.at[i, 'Score'] = response.document_sentiment.score
    organization_dataframe.at[i, 'Magnitude'] = response.document_sentiment.magnitude
    sleep(0.5)

print(organization_dataframe)
organization_dataframe.to_csv("survey_results.csv", encoding = 'ISO-8859-1')

gcs = storage.Client()
gcs.get_bucket(output_bucket).blob(output_csv_file).upload_from_filename('survey_results.csv', content_type='text/csv')
