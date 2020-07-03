from google.cloud import language_v1
from google.cloud.language_v1 import enums
from time import sleep
from tqdm import tqdm
import pandas as pd
import sys


def read_csv(csv_file):
    df = pd.read_csv(f"datasets/{csv_file}")
    df.fillna("", inplace=True)
    df.set_index(df.columns[0], inplace=True)
    df.index.name = "index"
    return df


def select_cols(dataframe, language='es', char='complete'):
    if language == 'es':
        if char == 'complete':
            df = dataframe[['missing_info', 'improvements']]
            return df
        if char == 'adj':
            df = dataframe[['missing_info' + '_' + char, 'improvements' + '_' + char]]
            return df
        if char == 'noun_adj':
            df = dataframe[['missing_info' + '_' + char, 'improvements' + '_' + char]]
            return df

    if language == 'en':
        if char == 'complete':
            df = dataframe[['missing_info' + '_' + language, 'improvements' + '_' + language]]
            return df
        if char == 'adj':
            df = dataframe[['missing_info' + '_' + language + '_' + char, 'improvements' + '_' + language + '_' + char]]
            return df
        if char == 'noun_adj':
            df = dataframe[['missing_info' + '_' + language + '_' + char, 'improvements' + '_' + language + '_' + char]]
            return df


def analyze_sentiment(text, client, language):
    """
    Analyzing Sentiment in text file stored in Cloud Storage
    Args:
      -text Text to analyze
      -client:
       e.g. language_v1.LanguageServiceClient()
      -language: 'es' or 'en'
       For list of supported languages:
       https://cloud.google.com/natural-language/docs/languages
    """
    type_ = enums.Document.Type.PLAIN_TEXT
    document = {"content": text, "type": type_, "language": language}
    encoding_type = enums.EncodingType.UTF8
    response = client.analyze_sentiment(document, encoding_type=encoding_type)
    return response


if __name__ == "__main__":
    # Get file name from arguments
    file_name = sys.argv[1]
    # Get which columns to process
    columns = sys.argv[2]

    # Get client from google api
    client = language_v1.LanguageServiceClient()

    
    # ['es', 'en']	
    for language in ['en']:
        # Read csv file
        data = read_csv(file_name)
        # Select columns to process
        data = select_cols(data, language, columns)
        data_cols = data.columns

        for col in data_cols:
            print(f"Working in language {language} with column {col}")

            # Loop through all rows by index
            for i in tqdm(data.index):
                response = analyze_sentiment(data.loc[i, col], client, language=language)
                data.loc[i, f"{col}_gscore"] = response.document_sentiment.score
                data.loc[i, f"{col}_gmagnitude"] = response.document_sentiment.magnitude
                sleep(0.5)

        data.drop(data_cols, axis=1, inplace=True)
        data.to_csv(f"datasets/gcloud_output_{language}_{columns}.csv")
        del data
    print("Finished!!")
