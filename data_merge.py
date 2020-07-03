from itertools import combinations
from tqdm import tqdm
import pandas as pd
import numpy as np
import pprint


def classify_sentiment(n_del, p_del, sent_score):
    """Function that classifies a sentiment score into one of three categories: 0 (negative), 1 (neutral),
    2 (positive) """
    if sent_score < n_del:
        return 0
    elif sent_score > p_del:
        return 2
    else:
        return 1


def get_error(data, column):
    improvements_err = (data["improvements_sentiment"] != data[column])
    return improvements_err.sum() / df.shape[0]


def get_best_delimiters(data, lang, app):
    column = f"improvements_{lang}_{app}"
    # All possible combinations for delimiters
    digits = np.arange(data[f"{column}_gscore"].min(), data[f"{column}_gscore"].max(), .01)
    delimiters = list(combinations(digits, 2))

    max_val = 0
    best_delimiter = delimiters[0]

    for delimiter in tqdm(delimiters):
        data[f"{column}_sentiment"] = data[f"{column}_gscore"].apply(lambda x: classify_sentiment(*delimiter, x))
        error = get_error(data, f"{column}_sentiment")
        if error > max_val:
            max_val = error
            best_delimiter = delimiter
    return best_delimiter, max_val


if __name__ == "__main__":
    df = pd.read_csv("datasets/dataset_oficial.csv", usecols=["Sentimiento 2"])
    df.columns = ["improvements_sentiment"]

    for app in ["complete", "noun_adj", "adj"]:
        for lang in ["es", "en"]:
            x_df = pd.read_csv(f"datasets/gcloud_output_{lang}_{app}_val.csv")
            x_df[f"improvements_{lang}_{app}_gscore"] = x_df[f"improvements_{lang}_{app}_gscore"] * \
                                                        (x_df[f"improvements_{lang}_{app}_gmagnitude"] + 1)
            y_df = pd.read_csv(f"datasets/gcloud_output_{lang}_{app}_prod.csv")
            y_df[f"improvements_{lang}_{app}_gscore"] = y_df[f"improvements_{lang}_{app}_gscore"] * \
                                                        (y_df[f"improvements_{lang}_{app}_gmagnitude"] + 1)
            z_df = pd.concat([x_df, y_df])
            z_df = z_df.set_index("index")

            df = df.merge(z_df[f"improvements_{lang}_{app}_gscore"], left_index=True, right_index=True)
            df.to_csv("datasets/complete_gscores.csv")

    results = dict()
    for app in ["complete", "noun_adj", "adj"]:
        for lang in ["es", "en"]:
            best_threshold, max_acc = get_best_delimiters(df, lang, app)
            results[f"improvements_{lang}_{app}"] = {}
            results[f"improvements_{lang}_{app}"]["best_threshold"] = best_threshold
            results[f"improvements_{lang}_{app}"]["max_accuracy"] = max_acc

    app = "complete"
    lang = "en"
    #column = f"improvements_{lang}_{app}"
    #df[f"{column}_sentiment"] = df[f"{column}_gscore"].apply(lambda x: classify_sentiment(-0.05266, 0.979148, x))
    #error = get_error(df, f"{column}_sentiment")

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(results)