import pandas as pd
import string
import re
from tqdm import tqdm
from nltk import FreqDist, download
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import stanfordnlp

# Run only once
# download('stopwords')
# download('punkt')
# stanfordnlp.download('es')   # This downloads the Spanish models for the neural pipeline
# stanfordnlp.download('en')   # This downloads the English models for the neural pipeline


def load_csv(file_name):
    dataframe = pd.read_csv(f"datasets/{file_name}.csv", parse_dates=["Fecha"], encoding='utf-8').drop_duplicates().fillna("")
    dataframe.columns = ["poll_id", "token", "institution", "is_info_useful", "is_info_missing", "missing_info",
                         "improvements", "date", "missing_info_sentiment", "improvements_sentiment",
                         "missing_info_en", "improvements_en"]
    dataframe["missing_info"] = dataframe["missing_info"].astype(str)
    dataframe["improvements"] = dataframe["improvements"].astype(str)
    dataframe["missing_info_en"] = dataframe["missing_info_en"].astype(str)
    dataframe["improvements_en"] = dataframe["improvements_en"].astype(str)
    dataframe.sort_values('date', ascending=True, inplace=True)
    na_filter = (dataframe["missing_info"] == "") & (dataframe["improvements"] == "")
    return dataframe[~na_filter]


def list_to_dict(lst):
    return {lst[i]: '' for i in range(0, len(lst), 2)}


def clean_txt(txt):
    punctuations = list_to_dict(list(string.punctuation))

    if txt in ["nan", "#VALUE!"]:
        return ""
    txt = txt.lower()

    # Remove tildes
    vocales = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u"}
    for vocal in vocales.keys():
        txt = txt.replace(vocal, vocales[vocal])

    # Remove e-mails: user@xxx.com
    email = '\S*@\S*\s?'
    pattern = re.compile(email)
    pattern.sub('', txt)

    # Remove urls
    re.sub(r'http\S+', '', txt)

    # Remove non-alphanumeric characters
    for punctuation in punctuations.keys():
        txt = txt.replace(punctuation, punctuations[punctuation])

    # Remove punctuation
    txt = re.sub(r'[^(a-zA-Z)\s]', '', txt)
    return txt


def remove_stop_words(string, language):
    if string == "":
        return ""
    # Create dictionary of stopwords for spanish and english
    stopwords_dict = {
        "es": list(set(stopwords.words('spanish'))),
        "en": list(set(stopwords.words('english')))
    }
    # tokenize
    tokenized = word_tokenize(string)
    # remove stopwords
    stopped = [w for w in tokenized if w not in stopwords_dict[language]]
    # join the list of above words to create a sentence without stop words
    filtered_string = " ".join(stopped)
    return filtered_string


def extract_pos(doc):
    # dictionary to hold english pos tags and their explanations
    pos_dict = {
        'CC': 'coordinating conjunction',
        'CD': 'cardinal digit',
        'DT': 'determiner',
        'EX': 'existential there (like: \"there is\" ... think of it like \"there exists\")',
        'FW': 'foreign word',
        'IN': 'preposition/subordinating conjunction',
        'JJ': 'adjective \'big\'',
        'JJR': 'adjective, comparative \'bigger\'',
        'JJS': 'adjective, superlative \'biggest\'',
        'LS': 'list marker 1)',
        'MD': 'modal could, will',
        'NN': 'noun, singular \'desk\'',
        'NNS': 'noun plural \'desks\'',
        'NNP': 'proper noun, singular \'Harrison\'',
        'NNPS': 'proper noun, plural \'Americans\'',
        'PDT': 'predeterminer \'all the kids\'',
        'POS': 'possessive ending parent\'s',
        'PRP': 'personal pronoun I, he, she',
        'PRP$': 'possessive pronoun my, his, hers',
        'RB': 'adverb very, silently,',
        'RBR': 'adverb, comparative better',
        'RBS': 'adverb, superlative best',
        'RP': 'particle give up',
        'TO': 'to go \'to\' the store.',
        'UH': 'interjection errrrrrrrm',
        'VB': 'verb, base form take',
        'VBD': 'verb, past tense took',
        'VBG': 'verb, gerund/present participle taking',
        'VBN': 'verb, past participle taken',
        'VBP': 'verb, sing. present, non-3d take',
        'VBZ': 'verb, 3rd person sing. present takes',
        'WDT': 'wh-determiner which',
        'WP': 'wh-pronoun who, what',
        'WP$': 'possessive wh-pronoun whose',
        'WRB': 'wh-abverb where, when',
        'QF': 'quantifier, bahut, thoda, kam (Hindi)',
        'VM': 'main verb',
        'PSP': 'postposition, common in indian langs',
        'DEM': 'demonstrative, common in indian langs'
    }
    parsed_text = {'word':[], 'pos':[], 'exp':[]}
    for sent in doc.sentences:
        for wrd in sent.words:
            if wrd.pos in pos_dict.keys():
                pos_exp = pos_dict[wrd.pos]
            else:
                pos_exp = 'NA'
            parsed_text['word'].append(wrd.text)
            parsed_text['pos'].append(wrd.pos)
            parsed_text['exp'].append(pos_exp)
    return pd.DataFrame(parsed_text)


def extract_adj_noun(txt, language="en", noun=False):
    if language == "es":
        corpus = nlp_es(txt)
    else:
        corpus = nlp_en(txt)
    BOW = extract_pos(corpus)
    adj_pos = ["ADJ", "JJ", "JJR", "JJS"]
    noun_pos = ["NOUN", "NN", "NNS", "NNP", "NNPS"]
    if noun:
        return " ".join(BOW[BOW["pos"].isin(adj_pos + noun_pos)]["word"])
    return " ".join(BOW[BOW["pos"].isin(adj_pos)]["word"])


if __name__ == "__main__":
    # Load oficial dataset
    df = load_csv("dataset_oficial")

    # Transform is_info_useful to boolean values
    df["is_info_useful"] = df["is_info_useful"].map({"Sí": 1, "No": 0, "": -1})
    df["is_info_missing"] = df["is_info_missing"].map({"Sí": 1, "No": 0, "": -1})

    # Clean text using regex
    df['missing_info'] = df['missing_info'].apply(lambda x: clean_txt(x) if len(x) > 1 else x)
    df['improvements'] = df['improvements'].apply(lambda x: clean_txt(x) if len(x) > 1 else x)
    df['missing_info_en'] = df['missing_info_en'].apply(lambda x: clean_txt(x) if len(x) > 1 else x)
    df['improvements_en'] = df['improvements_en'].apply(lambda x: clean_txt(x) if len(x) > 1 else x)

    # Separate institution column
    institution_split = df["institution"].str.split('-', 2, expand=True)
    df["institution"] = institution_split[0]
    df["institution_1"] = institution_split[1]
    df["institution_2"] = institution_split[2]

    # Instatiate tqdm with pandas in order to visualize progress (the following codes may last several minutes)
    tqdm.pandas()

    # Tokenizing and removing stopwords
    df['missing_info_wo_stop_words'] = df['missing_info'].progress_apply(lambda x: remove_stop_words(x, 'es') if len(x) > 1 else x)
    df['improvements_wo_stop_words'] = df['improvements'].progress_apply(lambda x: remove_stop_words(x, 'es') if len(x) > 1 else x)
    df['missing_info_en_wo_stop_words'] = df['missing_info_en'].progress_apply(lambda x: remove_stop_words(x, 'en') if len(x) > 1 else x)
    df['improvements_en_wo_stop_words'] = df['improvements_en'].progress_apply(lambda x: remove_stop_words(x, 'en') if len(x) > 1 else x)

    # NLTK Part of Speech (POS)

    # MODELS_DIR = "/home/omar/Documentos/S_AI/Equipo_4/stanford_tagger/stanfordnlp_resources"
    MODELS_DIR = "/home/julio/stanfordnlp_resources"

    # Set up a default neural pipeline
    nlp_es = stanfordnlp.Pipeline(lang='es', models_dir=MODELS_DIR, processors="tokenize,mwt,lemma,pos")
    nlp_en = stanfordnlp.Pipeline(lang='en', models_dir=MODELS_DIR, processors="tokenize,mwt,lemma,pos")

    # Create Bag of Words (BOW)
    # Get spanish adjectives and nouns + adjectives
    df["missing_info_adj"] = df["missing_info_wo_stop_words"].progress_apply(lambda x: extract_adj_noun(x, 'es') if len(x) > 1 else x)
    df["improvements_adj"] = df["improvements_wo_stop_words"].progress_apply(lambda x: extract_adj_noun(x, 'es') if len(x) > 1 else x)
    df["missing_info_noun_adj"] = df["missing_info_wo_stop_words"].progress_apply(lambda x: extract_adj_noun(x, 'es', True) if len(x) > 1 else x)
    df["improvements_noun_adj"] = df["improvements_wo_stop_words"].progress_apply(lambda x: extract_adj_noun(x, 'es', True) if len(x) > 1 else x)

    # Get english adjectives and nouns + adjectives
    df["missing_info_en_adj"] = df["missing_info_en_wo_stop_words"].progress_apply(lambda x: extract_adj_noun(x, 'en') if len(x) > 1 else x)
    df["improvements_en_adj"] = df["improvements_en_wo_stop_words"].progress_apply(lambda x: extract_adj_noun(x, 'en') if len(x) > 1 else x)
    df["missing_info_en_noun_adj"] = df["missing_info_en_wo_stop_words"].progress_apply(lambda x: extract_adj_noun(x, 'en', True) if len(x) > 1 else x)
    df["improvements_en_noun_adj"] = df["improvements_en_wo_stop_words"].progress_apply(lambda x: extract_adj_noun(x, 'en', True) if len(x) > 1 else x)

    # Reorder columns
    final_df = df[["date", "poll_id", "token", "institution", "institution_1", "institution_2", "is_info_useful",
                   "is_info_missing", "missing_info", "improvements", "missing_info_en", "improvements_en",
                   "missing_info_wo_stop_words", "improvements_wo_stop_words", "missing_info_en_wo_stop_words",
                   "improvements_en_wo_stop_words", "missing_info_adj", "improvements_adj", "missing_info_noun_adj",
                   "improvements_noun_adj", "missing_info_en_adj", "improvements_en_adj", "missing_info_en_noun_adj",
                   "improvements_en_noun_adj", "missing_info_sentiment", "improvements_sentiment"]]

    # Split dataframe by labeled sentiment
    validation_set = final_df.loc[final_df["missing_info_sentiment"] != ""]
    production_set = final_df.loc[final_df["missing_info_sentiment"] == ""]

    # Export data if everything went ok
    if validation_set.shape[0] + production_set.shape[0] == df.shape[0]:
        validation_set.to_csv("datasets/validation_set.csv")
        production_set.to_csv("datasets/production_set.csv")
