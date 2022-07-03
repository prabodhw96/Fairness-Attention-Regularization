# Main script for gathering args.
from model.data_loader import data_loader
from model.metrics import assess_performance_and_bias
from analysis import analyze_results
from argparse import ArgumentParser
import zipfile
from pathlib import Path
from transformers import BertForSequenceClassification, RobertaForSequenceClassification
import torch
import os
from model.classifier import train_factual_classifier, train_counterfactual_classifier
from model.layerwise_bias import compute_layerwise_bias_contribution
import wandb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
softmax = torch.nn.Softmax(dim=1).to(device)


def parse_args():
    """Parses the command line arguments."""
    parser = ArgumentParser()
    # choosing between our work and the baselines
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="The seed that we are running. We normally run every experiment for 5 seeds.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="learning rate for the classifier",
    )
    parser.add_argument(
        "--classifier_model",
        choices=[
            "bert-base-cased",
            "bert-base-uncased",
            "bert-large-cased",
            "bert-large-uncased",
            "roberta-base",
            "distilroberta-base",
            "distilbert-base-cased",
            "distilbert-base-uncased",
        ],
        default="bert-base-uncased",
        help="Type of classifier used",
    )
    parser.add_argument(
        "--dataset",
        choices=[
            "Jigsaw",
            "Wiki",
            "Twitter",
            "EEEC",
        ],
        default="Twitter",
        help="Type of dataset used",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of pretraining epochs for the classifier.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for the classifier.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="The maximum length of the sentences that we classify (in terms of the number of tokens)",
    )
    parser.add_argument(
        "--model_checkpoint_path",
        default="./saved_models/checkpoint-",
        help="Path to the saved classifier checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        default="output/",
        help="Directory to the output",
    )
    parser.add_argument(
        "--model_dir",
        default="saved_models/",
        help="Directory to saved models",
    )
    parser.add_argument(
        "--load_model_counterfactual",
        type=bool,
        default=False,
        help="Whether or not to load pre-trained model trained on the counterfactual data",
    ) 
    parser.add_argument(
        "--load_model_factual",
        type=bool,
        default=False,
        help="Whether or not to load pre-trained model trained on the factual data",
    )    
    parser.add_argument(
        "--load_layerwise_bias_contributions",
        type=bool,
        default=False,
        help="Whether or not to load pprecomputed layerwise bias contributions",
    )      
    parser.add_argument(
        "--use_wandb",
        type=bool,
        default=True,
        help="Whether or not to use wandb to visualize the results",
    )
    parser.add_argument(
        "--use_amulet",
        type=bool,
        default=False,
        help="Whether or not to run the code on Amulet, which is the cluster used at Microsoft research",
    )
    parser.add_argument(
        "--analyze_results",
        type=bool,
        default=False,
        help="Whether or not to analyze the results by computing the demographic parity after each regularization applied to the attention weights",
    )  
    parser.add_argument(
        "--analyze_attention",
        type=bool,
        default=False,
        help="Whether or not to analyze the attention map",
    )  
    parser.add_argument(
        "--num_tokens_logged",
        type=int,
        default=5,
        help="The value of k given that we log the top k tokens that the classification token attends to",
    )    
    parser.add_argument(
        "--reg_coeff",
        type=float,
        default=0.5,
        help="Regularization coefficient to be multiplied with the attention weights"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with zipfile.ZipFile("./bias_datasets.zip", "r") as zip_ref:
        zip_ref.extractall("./data")

    if args.use_wandb:
        wandb.init(
            name=str(args.dataset),
            project="Mitigating gender bias using causal inference",
            config=args,
        )

    train_dataset, val_dataset, test_dataset = data_loader(
        args.dataset,
        args.max_length,
        args.classifier_model,
    )
    
    model_dir = args.model_dir
    output_dir = args.output_dir
    
    if args.use_amulet:
        model_dir = f"{os.environ['AMLT_OUTPUT_DIR']}/" + model_dir
        output_dir = f"{os.environ['AMLT_OUTPUT_DIR']}/" + output_dir

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if args.classifier_model in [
        "bert-base-cased",
        "bert-large-cased",
        "distilbert-base-cased",
        "bert-base-uncased",
        "bert-large-uncased",
        "distilbert-base-uncased",
    ]:
        huggingface_model = BertForSequenceClassification
        model_type = "bert"
    elif args.classifier_model in ["roberta-base", "distilroberta-base"]:
        huggingface_model = RobertaForSequenceClassification
        model_type = "roberta"
        
    # Define pretrained tokenizer and model
    model_name = args.classifier_model
    model_factual_data = huggingface_model.from_pretrained(
        model_name,
        num_labels=len(set(train_dataset.labels)),
        output_attentions=args.analyze_attention,
    )

    model_factual_data = model_factual_data.to(device)
    
    model_counterfactual_data = huggingface_model.from_pretrained(
        model_name,
        num_labels=len(set(train_dataset.labels)),
        output_attentions=args.analyze_attention,
    )

    model_counterfactual_data = model_counterfactual_data.to(device)    
    
    if args.load_model_factual == False:
        # Train/Load the biased model using the factual data
        model_factual_data = train_factual_classifier(
            model_factual_data,
            args.dataset,
            args.max_length,
            args.classifier_model,
            args.batch_size,
            model_dir,
            args.use_amulet,
            args.num_epochs,
            args.learning_rate,
            args.seed,            
        )
    
        # save the best model trained on the factual data
        torch.save(
            model_factual_data.state_dict(),
            model_dir + args.classifier_model + "_" + args.dataset + "_factual.pt",
        )
    else:
        model_factual_data.load_state_dict(
            torch.load(
                model_dir + args.classifier_model + "_" + args.dataset + "_factual.pt" ,
                map_location=device,
            )
        )   
    
    if args.load_model_counterfactual == False:
        # Train the model using the counterfactual data
        model_counterfactual_data = train_counterfactual_classifier(
            model_counterfactual_data,
            args.dataset,
            args.max_length,
            args.classifier_model,
            args.batch_size,
            model_dir,
            args.use_amulet,
            args.num_epochs,
            args.learning_rate,
            args.seed,            
        )

        # save the best biased model
        torch.save(
            model_counterfactual_data.state_dict(),
            model_dir + args.classifier_model + "_" + args.dataset + "_counterfactual.pt",
        )
    else:
        model_counterfactual_data.load_state_dict(
            torch.load(
                model_dir + args.classifier_model + "_" + args.dataset + "_counterfactual.pt",
                map_location=device,
            )
        )       

    if args.load_layerwise_bias_contributions == False:
        layerwise_bias_contributions = compute_layerwise_bias_contribution(args.dataset,
            args.max_length,
            args.classifier_model,
            model_factual_data,
            model_counterfactual_data,        
            args.batch_size,
            args.reg_coeff
            )
    
        file_directory = output_dir + "analysis/"
    
        Path(file_directory).mkdir(parents=True, exist_ok=True)     
        
        torch.save(layerwise_bias_contributions, file_directory + args.classifier_model + "_" + args.dataset + '_layerwise_bias_contributions.pt')
        
    else:
        layerwise_bias_contributions = torch.load(file_directory + args.classifier_model + "_" + args.dataset + 'layerwise_bias_contributions.pt')

# Debiasing using attention regularization

    model_after_debiasing = huggingface_model.from_pretrained(
        model_name,
        num_labels=len(set(train_dataset.labels)),
        output_attentions=args.analyze_attention,
    )
    
    for layer_id in range(len(layerwise_bias_contributions)):
        
        model_after_debiasing.load_state_dict(
            torch.load(
                model_dir + args.classifier_model + "_" + args.dataset + "_factual.pt",
                map_location=device,
            )
        ) 
        model_after_debiasing = model_after_debiasing.to(device) 
        
        #start_idx, end_idx, layer_id = get_head_dim_idx(head_id, model_after_debiasing)
        for name, para in model_after_debiasing.named_parameters():
            if model_type + ".encoder.layer." + str(layer_id) + ".attention" in name:
              para.data *= args.reg_coeff 
              #para.data[start_idx: end_idx] *= args.reg_coeff        

        assess_performance_and_bias(
            layerwise_bias_contributions[layer_id],
            layer_id,
            model_factual_data,
            model_after_debiasing,
            args.dataset,
            args.max_length,
            args.classifier_model,
            output_dir,
            model_dir,
            args.batch_size,
            args.use_wandb,
        )
        if args.analyze_results:
            analyze_results(
                layerwise_bias_contributions[layer_id],
                layer_id,
                model_factual_data,
                model_after_debiasing,
                args.dataset,
                args.max_length,
                args.classifier_model,
                args.batch_size,
                output_dir,
                model_dir,
                args.analyze_attention,
                args.num_tokens_logged,
            )