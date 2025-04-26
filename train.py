from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from evaluate import load
import numpy as np
import wandb
from loguru import logger
from sklearn.metrics import classification_report
import pandas as pd


def compute_metrics(eval_pred):
    metrics = {}
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # Global metrics
    accuracy_metric = load("accuracy")
    f1_metric = load("f1")
    precision_metric = load("precision")
    recall_metric = load("recall")

    metrics.update(accuracy_metric.compute(predictions=predictions, references=labels))
    metrics.update(
        f1_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )
    )
    metrics.update(
        precision_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )
    )
    metrics.update(
        recall_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )
    )

    # Get per-class metrics
    class_report = classification_report(labels, predictions, output_dict=True)

    # Add per-class metrics to wandb
    for class_id in range(5):  # 5 classes (0 to 4)
        class_metrics = class_report[str(class_id)]
        metrics[f"class_{class_id}_precision"] = class_metrics["precision"]
        metrics[f"class_{class_id}_recall"] = class_metrics["recall"]
        metrics[f"class_{class_id}_f1"] = class_metrics["f1-score"]
        metrics[f"class_{class_id}_support"] = class_metrics["support"]

    return metrics


def main():
    # Initialize wandb
    wandb.init(project="amazon-reviews-es", name="distilbert-base-multilingual")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset("SetFit/amazon_reviews_multi_es")

    # Load tokenizer and model
    logger.info("Loading tokenizer and model...")
    model_name = "distilbert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=5  # 0 to 4 labels
    )

    # Tokenize function
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=512
        )
        # Include labels in the returned dictionary
        tokenized["labels"] = examples["label"]
        return tokenized

    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    # Remove all columns except 'text' and 'label'
    columns_to_remove = [
        col for col in dataset["train"].column_names if col not in ["text", "label"]
    ]
    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=columns_to_remove
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        run_name="distilbert-amazon-reviews",  # Add a distinct run name
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
        eval_strategy="steps",  # Changed from "epoch" to "steps"
        eval_steps=1000,  # Added: evaluate every 1000 steps
        save_strategy="steps",  # Changed to match evaluation_strategy
        save_steps=1000,  # Added: save every 1000 steps
        load_best_model_at_end=True,
        push_to_hub=False,
        report_to="wandb",
        logging_strategy="steps",  # Added: ensure logging matches evaluation
        logging_steps=1000,  # Added: log every 1000 steps
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics,
    )

    # Train the model
    logger.info("Starting training...")
    trainer.train()

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(tokenized_dataset["test"])

    # Log test results to wandb
    wandb.log({"test": test_results})

    # Save the model
    logger.info("Saving model...")
    trainer.save_model("./final_model")

    # Close wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
