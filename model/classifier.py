from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from model.data_loader import data_loader
from pathlib import Path
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_factual_classifier(
    model,
    dataset,
    max_length,
    classifier_model,
    batch_size,
    model_dir,
    use_amulet,
    num_epochs,
    learning_rate,
    seed,
):
    """
    args:
        dataset: the dataset used
        max_length: The maximum length of the sentences that we classify (in terms of the number of tokens)
        classifier_model: the model name
        batch_size: the batch size for the pretraiing (training the biase model)
        model_dir: the Directory to the model
        use_amulet: whether or not to run the code on Amulet, which is the cluster used at Microsoft research
        num_epochs: the number of epochs to train the biased model, which is done before bias mitigation
        learning_rate: the learning rate of the classifier
        seed: the seed used by the classifier
    returns:
        model: the model after pre-training
    """
    # Load the dataset
    train_dataset, val_dataset, test_dataset = data_loader(
        dataset,
        max_length,
        classifier_model,
    )
    # The number of epochs afterwhich we save the model.
    checkpoint_steps = int(train_dataset.__len__() / batch_size)

    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # Define Trainer parameters
    classifier_args = TrainingArguments(
        output_dir=model_dir,
        evaluation_strategy="steps",
        eval_steps=checkpoint_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        seed=seed,
        load_best_model_at_end=True,
    )
    
    # Define Trainer
    trainer = Trainer(
        model=model,
        args=classifier_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train pre-trained model
    trainer.train()

    return model


def train_counterfactual_classifier(
    model,
    dataset,
    max_length,
    classifier_model,
    batch_size,
    model_dir,
    use_amulet,
    num_epochs,
    learning_rate,
    seed,
):
    """
    Train a classifier to be used as our starting point for polcy gradient.
    We can either train from scratch or load a pretrained model depending on
    the user's choice.
    args:
        dataset: the dataset used
        max_length: The maximum length of the sentences that we classify (in terms of the number of tokens)
        classifier_model: the model name
        batch_size: the batch size for the pretraiing (training the biase model)
        model_dir: the Directory to the model
        use_amulet: whether or not to run the code on Amulet, which is the cluster used at Microsoft research
        num_epochs: the number of epochs to train the biased model, which is done before bias mitigation
        learning_rate: the learning rate of the classifier
        seed: the seed used by the classifier
    returns:
        model: the model after pre-training
    """
    # Load the dataset
    train_dataset, val_dataset, test_dataset = data_loader(
        dataset,
        max_length,
        classifier_model,
        apply_gender_flipping=True,
    )
    # The number of epochs afterwhich we save the model.
    checkpoint_steps = int(train_dataset.__len__() / batch_size)

    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # Define Trainer parameters

    # Define Trainer
    classifier_args = TrainingArguments(
        output_dir=model_dir,
        evaluation_strategy="steps",
        eval_steps=checkpoint_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        seed=seed,
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=classifier_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train pre-trained model
    trainer.train()

    return model