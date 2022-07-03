import torch
import numpy as np
from model.data_loader import data_loader



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
softmax = torch.nn.Softmax(dim=1).to(device)

#def get_head_dim_idx(head, model):
#  layer = int(head/model.config.num_attention_heads)
#  head_dim = int(model.config.hidden_size/model.config.num_attention_heads)
#  start = head_dim*(head - layer*model.config.num_attention_heads)
#  end = head_dim*(head - (layer*model.config.num_attention_heads) + 1)
#  return start, end, layer

def compute_ATE(
    split_dataset,
    model_factual_data,
    model_counterfactual_data,
    batch_size,
    classifier_model,
    attention_intevention_layers = None,

):
    """num_labels
    Compute the ATE.
    args:
        split_dataset: the dataset split object on which the metrics are measured.
        model_factual: the model before updating its weights for bias reduction
        model_counterfactual: the model after updating its weights for bias reduction
        batch_size: the size of the batch used
    returns:
        the function returns the ATE.
    """
    if classifier_model in [
        "bert-base-cased",
        "bert-large-cased",
        "distilbert-base-cased",
        "bert-base-uncased",
        "bert-large-uncased",
        "distilbert-base-uncased",
    ]:
        model_type = "bert"
    elif classifier_model in ["roberta-base", "distilroberta-base"]:
        model_type = "roberta"
         
    with torch.no_grad():


        if attention_intevention_layers != None:
            for attention_intevention_layer in attention_intevention_layers:
                for name_factual, para_factual in model_factual_data.named_parameters():
                  for name_counterfactual, para_counterfactual in model_counterfactual_data.named_parameters():
#                    start_idx, end_idx, layer_id = get_head_dim_idx(attention_intevention_head, model_factual_data)
                    if (model_type + ".encoder.layer." + str(attention_intevention_layer) + ".attention" in name_factual) and (model_type + ".encoder.layer." + str(attention_intevention_layer) + ".attention" in name_counterfactual) and (name_factual == name_counterfactual):
                      #print(layer_id, start_idx, end_idx, name_factual)
                      para_counterfactual.data = para_factual.data
#                      para_counterfactual.data[start_idx: end_idx] = para_factual.data[start_idx: end_idx]            

        y_pred_counterfactual = torch.ones([0, len(set(split_dataset.labels))]).to(device)
        y_pred_factual = torch.ones([0, len(set(split_dataset.labels))]).to(device)

        for i in range(int(np.ceil(len(split_dataset) / batch_size))):

            results_counterfactual = model_counterfactual_data.forward(
                input_ids=torch.tensor(
                    split_dataset.encodings["input_ids"][
                        i * batch_size : (i + 1) * batch_size
                    ]
                ).to(device),
                attention_mask=torch.tensor(
                    split_dataset.encodings["attention_mask"][
                        i * batch_size : (i + 1) * batch_size
                    ]
                ).to(device),
            )[0]

            results_factual = model_factual_data.forward(
                input_ids=torch.tensor(
                    split_dataset.encodings["input_ids"][
                        i * batch_size : (i + 1) * batch_size
                    ]
                ).to(device),
                attention_mask=torch.tensor(
                    split_dataset.encodings["attention_mask"][
                        i * batch_size : (i + 1) * batch_size
                    ]
                ).to(device),
            )[0]

            # Add them to the total predictions
            y_pred_counterfactual = torch.cat(
                (y_pred_counterfactual, results_counterfactual), 0
            )

            # Add them to the total predictions
            y_pred_factual = torch.cat(
                (y_pred_factual, results_factual), 0
            )
        # ===================================================#
        # Here we calculate the accuracy
        # ATE = torch.mean(torch.abs(torch.norm(y_pred_factual[:,1] - y_pred_counterfactual, dim=1, p=2)))
        # This is assuming that we have binary classification
        ATE = torch.mean(y_pred_factual[:,1]) - torch.mean(y_pred_counterfactual[:,1])

        return ATE

def compute_layerwise_bias_contribution(
    dataset_name,
    max_length,
    classifier_model,
    model_factual_data,
    model_counterfactual_data,
    batch_size,
    reg_coeff
):
    """
    args:
        dataset_name: the name of the dataset used
        model_factual: the model before updating its weights for bias reduction
        model_counterfactual: the model after updating its weights for bias reduction
        batch_size: the size of the batch used
        num_labels: the numnber of labels in the dataset
    returns:
        the function returns the layerwise bias contribution
    """
    layerwise_bias_contribution = []

    train_dataset, val_dataset, test_dataset = data_loader(
        dataset_name,
        max_length,
        classifier_model,
    )
    
    bias_all_layers = compute_ATE(test_dataset, 
                                             model_factual_data,
                                             model_counterfactual_data,
                                             batch_size,
                                             classifier_model,
                                             attention_intevention_layers = None)

#    for i in range(model_factual_data.config.num_hidden_layers * model_factual_data.config.num_attention_heads): 
    for i in range(model_factual_data.config.num_hidden_layers): 
        print(i)
        bias_all_layers_except_i = compute_ATE(test_dataset, 
                                             model_factual_data,
                                             model_counterfactual_data,
                                             batch_size,
                                             classifier_model,
                                             attention_intevention_layers = [i])
        
        layerwise_bias_contribution.append(bias_all_layers - bias_all_layers_except_i)
        
    return layerwise_bias_contribution