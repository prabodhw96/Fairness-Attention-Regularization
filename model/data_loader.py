import pandas as pd
from transformers import BertTokenizer, RobertaTokenizer
import torch


# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        encodings,
        labels=None,
    ):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def data_loader(
    dataset,
    max_length,
    classifier_model,
    apply_gender_flipping=None,
    IPTTS=False,
):
    """
    Load the data  from the CSV files and an object for each split in the dataset.
    For each dataset, we have the data stored in both the original form, as well
    the gender flipped form.
    args:
        dataset: the dataset used
        max_length: The maximum length of the sentences that we classify (in terms of the number of tokens)
        classifier_model: the model name
        apply_gender_flipping: a flag to choose whether or not to apply gender-flipping to the data
        IPTTS: a flag that means that we want to also return the IPTTS dataset,
            which is the dataset on which we compute the bias metrics. We can also
            compute the AUC score on it.
    returns:
        the function returns 3 objects, for the training, validation and test
        datasets. Each object contains the tokenized data and the corresponding
        labels.
    """
    data_train = pd.read_csv("./data/" + dataset + "_train_original_gender.csv")
    data_valid = pd.read_csv("./data/" + dataset + "_valid_original_gender.csv")
    data_test = pd.read_csv("./data/" + dataset + "_test_original_gender.csv")

    # The gender swap means that we flip the gender in each example in out dataset.
    # For example, the sentence "he is a doctor" becomes "she is a doctor".
    data_train_gender_swap = pd.read_csv("./data/" + dataset + "_train_gender_swap.csv")
        
    if apply_gender_flipping:
        data_train = data_train_gender_swap        

    model_name = classifier_model
    if model_name in [
        "bert-base-cased",
        "bert-large-cased",
        "distilbert-base-cased",
        "bert-base-uncased",
        "bert-large-uncased",
        "distilbert-base-uncased",
    ]:
        tokenizer = BertTokenizer.from_pretrained(
            "./saved_models/cached_tokenizers/" + model_name
        )
    elif model_name in ["roberta-base", "distilroberta-base"]:
        tokenizer = RobertaTokenizer.from_pretrained(
            "./saved_models/cached_tokenizers/" + model_name
        )

    # ----- 1. Preprocess data -----#
    # Preprocess data
    X_train = list(data_train[data_train.columns[0]])
    y_train = list(data_train[data_train.columns[1]])

    X_val = list(data_valid[data_valid.columns[0]])
    y_val = list(data_valid[data_valid.columns[1]])

    X_test = list(data_test[data_test.columns[0]])
    y_test = list(data_test[data_test.columns[1]])


    X_train_tokenized = tokenizer(
        X_train, padding=True, truncation=True, max_length=max_length
    )
    X_val_tokenized = tokenizer(
        X_val, padding=True, truncation=True, max_length=max_length
    )
    X_test_tokenized = tokenizer(
        X_test, padding=True, truncation=True, max_length=max_length
    )

    train_dataset = Dataset(
        encodings=X_train_tokenized,
        labels=y_train,
    )
    val_dataset = Dataset(
        encodings=X_val_tokenized,
        labels=y_val,
    )
    test_dataset = Dataset(
        encodings=X_test_tokenized,
        labels=y_test,
    )

    # IPTTS is a synthetic dataset that is used to compute the fairness metrics
    if IPTTS:
        data_IPTTS_gender = pd.read_csv("./data/" + "madlib.csv")
        X_IPTTS_gender = list(data_IPTTS_gender[data_IPTTS_gender.columns[0]])
        y_IPTTS_gender = list(data_IPTTS_gender["Class"])
        X_IPTTS_gender_tokenized = tokenizer(
            X_IPTTS_gender, padding=True, truncation=True, max_length=max_length
        )
        IPTTS_gender_dataset = Dataset(
            encodings=X_IPTTS_gender_tokenized,
            labels=y_IPTTS_gender,
        )

        data_IPTTS_social = pd.read_csv("./data/" + "bias_madlibs_77k.csv")
        X_IPTTS_social = list(data_IPTTS_social[data_IPTTS_social.columns[0]])
        y_IPTTS_social = list(data_IPTTS_social["Class"])
        X_IPTTS_social_tokenized = tokenizer(
            X_IPTTS_social, padding=True, truncation=True, max_length=max_length
        )
        IPTTS_social_dataset = Dataset(
            encodings=X_IPTTS_social_tokenized,
            labels=y_IPTTS_social,
        )

        return (
            train_dataset,
            val_dataset,
            test_dataset,
            IPTTS_gender_dataset,
            IPTTS_social_dataset,
        )

    return train_dataset, val_dataset, test_dataset
