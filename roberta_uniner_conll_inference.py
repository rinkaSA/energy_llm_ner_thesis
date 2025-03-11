import argparse
import time
import yaml
import json
import torch
from transformers import pipeline
from datasets import load_dataset
import mlflow
import os
import matplotlib.pyplot as plt
import evaluate
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import sys
sys.path.insert(0, '/data/horse/ws/irve354e-uniNer_test/code/universal-ner/src')
from utils import preprocess_instance 

load_dotenv()

def np_encoder(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def compute_token_offsets(tokens, joined_text):
    """Compute character offsets for each token in the joined text."""
    offsets = []
    current_pos = 0
    for token in tokens:
        start = joined_text.find(token, current_pos)
        if start == -1:
            start = current_pos
        end = start + len(token)
        offsets.append((start, end))
        current_pos = end + 1
    return offsets


def build_input(sample, model_type):
    """
    Build model input from a sample.
    For Roberta, simply return the joined text.
    For UniversalNER, build a conversation prompt that instructs the model
    to output a JSON list of BIO tags corresponding exactly to the tokens.
    """
    if model_type == "roberta":
        return sample["joined_text"]
    elif model_type == "universalner":
        # Use conversation-based format for UniversalNER.
        conversation = [
            {"from": "human", "value": f"Text: {sample['joined_text']}"},
            {"from": "human", "value": f"Tokens: {sample['tokens']}"},
            {"from": "human", "value": "I have read the text."},
            {"from": "human", "value": "Please output a JSON list of BIO tags corresponding exactly to the tokens above."},
        ]
        # Import your preprocess_instance function from utils.
        from utils import preprocess_instance  
        prompt = preprocess_instance(conversation)
        return prompt
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def parse_roberta_output(sample, output):
    """
    Parse the output from a Roberta NER pipeline.
    The output is a list of entity predictions; we use token offsets to convert these into a list of BIO tags.
    """
    offsets = compute_token_offsets(sample["tokens"], sample["joined_text"])
    predicted_labels = ["O"] * len(sample["tokens"])
    for entity in output:
        entity_start = entity["start"]
        entity_end = entity["end"]
        first_token = True
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start >= entity_start and tok_end <= entity_end:
                if first_token:
                    predicted_labels[i] = "B-" + entity["entity_group"]
                    first_token = False
                else:
                    predicted_labels[i] = "I-" + entity["entity_group"]
    return predicted_labels


def parse_universalner_output(sample, output):
    """
    Parse the output from a UniversalNER text-generation pipeline.
    We expect the generated text to be a JSON list of BIO tags.
    """
    generated_text = output.get("generated_text", "")
    try:
        pred_labels = json.loads(generated_text)
        if not isinstance(pred_labels, list) or len(pred_labels) != len(sample["tokens"]):
            raise ValueError("Length mismatch or invalid format")
    except Exception as e:
        print(f"Error parsing output for sample: {e}")
        pred_labels = ["O"] * len(sample["tokens"])
    return pred_labels


def evaluate_and_log(all_true_labels, all_predicted_labels, comparisons, artifact_prefix):
    """
    Compute seqeval metrics, log overall and per-entity scores,
    save detailed JSON results, and generate/log a confusion matrix.
    """
    seqeval_metric = evaluate.load("seqeval")
    results = seqeval_metric.compute(predictions=all_predicted_labels, references=all_true_labels)

    mlflow.log_metric("overall_precision", results["overall_precision"])
    mlflow.log_metric("overall_recall", results["overall_recall"])
    mlflow.log_metric("overall_f1", results["overall_f1"])
    mlflow.log_metric("overall_accuracy", results["overall_accuracy"])

    detailed_results = {"comparisons": comparisons, "evaluation": results}
    detailed_json = f"{artifact_prefix}_detailed_results.json"
    with open(detailed_json, "w") as f:
        json.dump(detailed_results, f, indent=2, default=np_encoder)
    mlflow.log_artifact(detailed_json)

    true_flat = [label for seq in all_true_labels for label in seq]
    pred_flat = [label for seq in all_predicted_labels for label in seq]
    unique_labels = sorted(list(set(true_flat + pred_flat)))
    cm = confusion_matrix(true_flat, pred_flat, labels=unique_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation="vertical")
    plt.title("Confusion Matrix for NER Predictions")
    plt.tight_layout()
    cm_filename = f"{artifact_prefix}_confusion_matrix.png"
    plt.savefig(cm_filename)
    plt.close()
    mlflow.log_artifact(cm_filename)

    return results


def run_ner(args):
    """
    Unified function to run NER evaluation for either a Roberta-based or UniversalNER model.
    Both pipelines are processed in batches.
    """
    mlflow.set_experiment(args.experiment_name)
    run_name = (
        args.run_name
        if args.run_name != "default_run"
        else f"NER_{args.model_path.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    with mlflow.start_run(run_name=run_name, log_system_metrics=True):
        mlflow.log_param("experiment_name", args.experiment_name)
        mlflow.log_param("run_name", run_name)
        mlflow.log_param("model_path", args.model_path)
        mlflow.log_param("dataset_name", args.dataset_name)
        mlflow.log_param("model_type", args.model_type)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("max_new_tokens", args.max_new_tokens)

        dataset = load_dataset(args.dataset_name)['test']
        label_names = dataset.features["ner_tags"].feature.names

        samples = []
        for sample in dataset:
            tokens = sample["tokens"]
            joined_text = " ".join(tokens)
            true_labels = [label_names[tag] for tag in sample["ner_tags"]]
            samples.append({
                "tokens": tokens,
                "joined_text": joined_text,
                "true_labels": true_labels
            })

        if args.model_type == "roberta":
            model_pipeline = pipeline(
                "ner",
                model=args.model_path,
                tokenizer=args.model_path,
                aggregation_strategy="simple"
            )
        elif args.model_type == "universalner":
            model_pipeline = pipeline(
                "text-generation",
                model=args.model_path,
                torch_dtype=torch.float16,
                device=0
            )
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}")


        effective_batch_size = len(samples) if args.use_full_batch else args.batch_size
        all_true_labels = [sample["true_labels"] for sample in samples]
        all_predicted_labels = []
        comparisons = []
        num_samples = len(samples)
        start_time = time.time()

        for i in range(0, num_samples, effective_batch_size):
            batch = samples[i : i + effective_batch_size]
            inputs = [build_input(sample, args.model_type) for sample in batch]
            if args.model_type == "roberta":
                outputs = model_pipeline(inputs)
            elif args.model_type == "universalner":
                outputs = model_pipeline(inputs, max_length=args.max_new_tokens, return_full_text=False)
            else:
                print('wrong input model')
                outputs = []

            # For the "ner" pipeline (Roberta), if a batch was provided, outputs is a list of lists.
            if args.model_type == "roberta":
                if outputs and isinstance(outputs[0], dict):
                    outputs = [outputs]
            for sample, output in zip(batch, outputs):
                if args.model_type == "roberta":
                    pred_labels = parse_roberta_output(sample, output)
                elif args.model_type == "universalner":
                    pred_labels = parse_universalner_output(sample, output)
                else:
                    pred_labels = ["O"] * len(sample["tokens"])
                all_predicted_labels.append(pred_labels)
                comparisons.append({
                    "tokens": sample["tokens"],
                    "true_labels": sample["true_labels"],
                    "predicted_labels": pred_labels,
                    "input": build_input(sample, args.model_type),
                    "raw_output": output
                })

        total_time = time.time() - start_time
        mlflow.log_metric("total_processing_time", total_time)
        print(f"Total processing time: {total_time:.2f} seconds")

        results = evaluate_and_log(all_true_labels, all_predicted_labels, comparisons, artifact_prefix=args.model_type)
        print("Evaluation results:")
        print(json.dumps(results, indent=2, default=np_encoder))
        print(mlflow.active_run().info)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file with parameters")
    parser.add_argument("--experiment_name", type=str, default="default_experiment", help="MLflow experiment name")
    parser.add_argument("--run_name", type=str, default="default_run", help="MLflow run name")
    parser.add_argument("--model_path", type=str, default="Universal-NER/UniNER-7B-type", help="Model path for NER")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens to generate (for UniversalNER)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--use_full_batch", action="store_true",
                        help="If set and the dataset is small, process the entire dataset as one batch.")
    parser.add_argument("--model_type", type=str, choices=["roberta", "universalner"], default="universalner",
                        help="Which NER model to run")
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            setattr(args, key, value)

    print("Final configuration:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    run_ner(args)


if __name__ == "__main__":
    main()