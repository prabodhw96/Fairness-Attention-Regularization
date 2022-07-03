from model.data_loader import data_loader
import pandas as pd
from utils import log_topk_attention_tokens
from transformers import BertTokenizer, RobertaTokenizer
import torch
import re
import numpy as np
from os.path import exists


# from gender_bender import gender_bend
from pathlib import Path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
softmax = torch.nn.Softmax(dim=1).to(device)
                
def analyze_results(
    layer_bias,
    layer_id,
    model_before_debiasing,
    model_after_debiasing,        
    dataset,
    max_length,
    classifier_model,
    batch_size,
    output_dir,
    model_dir,
    analyze_attention,
    num_tokens_logged,
):
    """
    Analyze the results on the validation data by focusing on:
    1) Attention weights: We log the top k tokens to which the classification token (CLS) attends before and after de-biasing.
    2) Type of examples: We follow the procedure in https://arxiv.org/pdf/2009.10795.pdf where the examples
        are categorized into "easy-to-learn", "hard-to-learn" and "ambiguous". The intuition is to know which category is mostly affected by the de-biasing algorithm.
    args:
        dataset: the dataset used
        max_length: The maximum length of the sentences that we classify (in terms of the number of tokens)
        classifier_model: the model name
        num_epochs_pretraining: the number of epochs for the pretraining (training the biased model)
        batch_size: the batch size for the pretraiing (training the biased model)
        output_dir: the output directory that contains the results
        model_dir: the Directory to the model
        batch_size: the batch size for the training of the debiased model
        analyze_attention: whether or not to copmute the distribution of the attention weights
        num_tokens_logged: the top k tokens  that we consider, which the CLS tokens attends to
        method: the debiasing method used
    returns:
        the function doesnt return anything, since the output is written in a csv file.
    """
    train_dataset, val_dataset, test_dataset = data_loader(
        dataset,
        max_length,
        classifier_model,
    )

    file_directory = output_dir + "analysis/"
    Path(file_directory).mkdir(parents=True, exist_ok=True)    

    for split in ["valid", "test"]:
        file_path = file_directory + split + "_data_analysis_" + dataset + "_" + classifier_model + ".csv"
        #file_directory_exists = exists(file_path)        
        if layer_id != 0:
            # If this is the first layer, we load the original csv file. After that, we load the last one we saved.
            data = pd.read_csv(file_path)
        else:
            data = pd.read_csv("./data/" + dataset + "_" + split + "_original_gender.csv")
            
        if split == "valid":
            dataset_split = val_dataset
        elif split == "test":
            dataset_split = test_dataset

        if classifier_model in [
            "bert-base-cased",
            "bert-large-cased",
            "distilbert-base-cased",
            "bert-base-uncased",
            "bert-large-uncased",
            "distilbert-base-uncased",
        ]:
            tokenizer = BertTokenizer.from_pretrained(
                "./saved_models/cached_tokenizers/" + classifier_model
            )
        elif classifier_model in ["roberta-base", "distilroberta-base"]:
            tokenizer = RobertaTokenizer.from_pretrained(
                "./saved_models/cached_tokenizers/" + classifier_model
            )

        number_of_labels = len(set(dataset_split.labels))
        prediction_before_debiasing = torch.ones([0, number_of_labels]).to(device)
        for i in range(int(np.ceil(len(dataset_split) / batch_size))):
            with torch.no_grad():
                prediction = model_before_debiasing.forward(
                    input_ids=torch.tensor(
                        dataset_split.encodings["input_ids"][
                            i * batch_size : (i + 1) * batch_size
                        ]
                    ).to(device),
                    attention_mask=torch.tensor(
                        dataset_split.encodings["attention_mask"][
                            i * batch_size : (i + 1) * batch_size
                        ]
                    ).to(device),
                )["logits"]

            predictions_batch = softmax(
                torch.cat(
                    [
                        torch.unsqueeze(prediction[j], dim=0)
                        for j in range(len(prediction))
                    ]
                )
            ).to(device)
            prediction_before_debiasing = torch.cat(
                (prediction_before_debiasing, predictions_batch), 0
            ).to(device)

        y_pred_before_debiasing = torch.argmax(
            prediction_before_debiasing,
            axis=1,
        )
        # =======================================================
        prediction_after_debiasing = torch.ones([0, number_of_labels]).to(device)
        for i in range(int(np.ceil(len(dataset_split) / batch_size))):
            with torch.no_grad():
                prediction = model_after_debiasing.forward(
                    input_ids=torch.tensor(
                        dataset_split.encodings["input_ids"][
                            i * batch_size : (i + 1) * batch_size
                        ]
                    ).to(device),
                    attention_mask=torch.tensor(
                        dataset_split.encodings["attention_mask"][
                            i * batch_size : (i + 1) * batch_size
                        ]
                    ).to(device),
                )["logits"]

            predictions_batch = softmax(
                torch.cat(
                    [
                        torch.unsqueeze(prediction[j], dim=0)
                        for j in range(len(prediction))
                    ]
                )
            ).to(device)
            prediction_after_debiasing = torch.cat(
                (prediction_after_debiasing, predictions_batch), 0
            ).to(device)
        # Get the output of the model after debiasing
        y_pred_after_debiasing = torch.argmax(prediction_after_debiasing, axis=1)

        if analyze_attention:
            data = log_topk_attention_tokens(
                batch_size,
                num_tokens_logged,
                data,
                model_before_debiasing,
                model_after_debiasing,
                dataset_split,
                tokenizer,
            )


        # To analyze our results, we keep track of the confidence and variability in prediction of each example in the validation data, as well as whether or not
        # it is correctly classified before and after de-biasing.
        ground_truth_labels = torch.tensor(dataset_split.labels).to(device)
        data["Correct classification? before debiasing"] = (
            ground_truth_labels.cpu() == y_pred_before_debiasing.cpu()
        )
        data["Correct classification? after debiasing"] = (
            ground_truth_labels.cpu() == y_pred_after_debiasing.cpu()
        )

        for k in range(prediction_before_debiasing.shape[1]):
            if layer_id == 0:
                data["number of tokens"] = data[data.columns[0]].apply(
                    lambda x: len(re.findall(r"\w+", x))
                )                
            # To avoid repetition, we only save the DP before debiasing when logging the results for layer_id = 0
                data["p(y=1|x) our model before debiasing for label " + str(k) + " layer " + str(layer_id)] = list(
                    prediction_before_debiasing[:, k].cpu().detach().numpy()
                )
            data["p(y=1|x) our model after debiasing for label " + str(k) + " layer " + str(layer_id)] = list(
                prediction_after_debiasing[:, k].cpu().detach().numpy()
            )
            
        data["Bias contribution for layer " + str(layer_id)] = layer_bias.cpu().detach().numpy()
                    
            
        data.to_csv(
            file_directory
            + split
            + "_data_analysis_"
            + dataset
            + "_"
            + classifier_model
            + ".csv",
            index=False,
        )

