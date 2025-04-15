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


def compute_metrics(eval_pred):
    metrics = {}
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # Load and compute metrics
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
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=512
        )

    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=dataset["train"].column_names
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        report_to="wandb",
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
