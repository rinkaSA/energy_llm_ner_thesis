import argparse
import json
import sys
import time
from datetime import datetime

import evaluate
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import plotly.graph_objects as go
import torch
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from transformers import pipeline

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


def build_input(sample, model_type, entity_type=None):
    """
    Build model input from a sample.
    For Roberta, simply return the joined text.
    For UniversalNER, build a conversation prompt that instructs the model
    to output a JSON list of BIO tags corresponding exactly to the tokens.
    """
    if model_type == "roberta":
        return sample["joined_text"]
    elif model_type == "universalner":
        #conversation = [
        #    {"from": "human", "value": f"Text: {sample['joined_text']}"},
        #    {"from": "human", "value": f"Tokens: {sample['tokens']}"},
         #   {"from": "human", "value": "I have read the text."},
         #   {"from": "human", "value": "Please output a JSON list of BIO tags corresponding exactly to the tokens above."},
        #]
        conversation = {"conversations": [{"from": "human", "value": f"Text: {sample['joined_text']}"}, {"from": "gpt", "value": "I've read this text."}, {"from": "human", "value": f"What describes {entity_type} in the text?"}, {"from": "gpt", "value": "[]"}]}

      
        prompt = preprocess_instance(conversation["conversations"])
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


def parse_universalner_output(sample, output, entity_type):
    """
    Parse the output from a UniversalNER text-generation pipeline.
    The output is expected to be a JSON list of entity phrases (words or phrases) for the specified type.
    This function produces token-level BIO labels corresponding to sample["tokens"]:
      - Tokens that belong to a predicted entity are labeled as B-<entity_type> (first token) or I-<entity_type> (subsequent tokens).
      - All other tokens are labeled as "O".
    """
    generated_text = output[0].get("generated_text", "")
    try:
        predicted_entities = json.loads(generated_text)
        if not isinstance(predicted_entities, list):
            raise ValueError("Predicted entities is not a list")
    except Exception as e:
        print(f"Error parsing output for sample: {e}")
        predicted_entities = []
    
    tokens = sample["tokens"]
    labels = ["O"] * len(tokens)
    
    entity_found = False
    for entity in predicted_entities:
        entity_found =  True
        if " " in entity:
            entity_tokens = entity.split()
            n = len(entity_tokens)
            for i in range(len(tokens) - n + 1):
                if tokens[i:i+n] == entity_tokens:
                    labels[i] = f"B-{entity_type}"
                    for j in range(1, n):
                        labels[i+j] = f"I-{entity_type}"
                    
        else:
            for i, token in enumerate(tokens):
                if token == entity:
                    if i == 0 or labels[i-1] == "O":
                        labels[i] = f"B-{entity_type}"
                    else:
                        labels[i] = f"I-{entity_type}"
                    

    if not entity_found:
        labels = ["O"] * len(tokens)
    
    return labels


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
    overall_metrics = {}

    for key, value in results.items():
        if key.startswith("overall_"):
            mlflow.log_metric(key, value)
            overall_metrics[key] = value 
    detailed_results = {"comparisons": comparisons, "evaluation": results}
    detailed_json = f"/data/horse/ws/irve354e-uniNer_test/energy_ner_llm/energy_ner_llm/reports/{artifact_prefix}_detailed_results.json"
    with open(detailed_json, "w") as f:
        json.dump(detailed_results, f, indent=2, default=np_encoder)
    mlflow.log_artifact(detailed_json)

    
    cm_filename = f"/data/horse/ws/irve354e-uniNer_test/energy_ner_llm/energy_ner_llm/reports/figures/{artifact_prefix}_confusion_matrix.png"
    true_flat = [label for seq in all_true_labels for label in seq]
    pred_flat = [label for seq in all_predicted_labels for label in seq]
    unique_labels = sorted(list(set(true_flat + pred_flat)))
    for entity_type in unique_labels:
        true_binary = [entity_type if label == entity_type else 'Other' for label in true_flat]
        pred_binary = [entity_type if label == entity_type else 'Other' for label in pred_flat]

        cm = confusion_matrix(true_binary, pred_binary, labels=[entity_type, 'Other'])
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=[entity_type, 'Other'])
        disp.plot(ax=ax, cmap=plt.cm.Blues, values_format=".2%")
        plt.title(f"Confusion Matrix for Entity Type: {entity_type}")
        plt.tight_layout()

        per_type_filename = cm_filename.replace("confusion_matrix", f"{entity_type}_confusion_matrix")
        plt.savefig(per_type_filename)
        plt.close()
        mlflow.log_artifact(per_type_filename)

    #plot_all_entity_confusion_matrices(all_true_labels, all_predicted_labels, unique_labels)
    #mlflow.log_artifact(cm_filename)

    return results


def plot_all_entity_confusion_matrices(true_labels_list, pred_labels_list, entity_types):
    """
    For each entity type in entity_types, create an interactive Plotly confusion matrix.
    For each token in the flattened lists, a token is considered positive if its label is
    either 'B-<entity_type>' or 'I-<entity_type>', and negative otherwise.
    
    :param true_labels_list: List of lists of true BIO labels.
    :param pred_labels_list: List of lists of predicted BIO labels.
    :param entity_types: List of entity types to evaluate (e.g. ["PER", "LOC", "ORG"]).
    :return: A dictionary mapping each entity type to a tuple (fig, cm_percent),
             where fig is the Plotly Figure and cm_percent is the 2D array (in percentage).
    """
    all_figures = {}
    all_cm_percent = {}
    
    for et in entity_types:
        def to_binary(label):
            return 1 if label in {f"B-{et}", f"I-{et}"} else 0

        true_flat = [label for seq in true_labels_list for label in seq]
        pred_flat = [label for seq in pred_labels_list for label in seq]
        
        true_binary = [to_binary(lbl) for lbl in true_flat]
        pred_binary = [to_binary(lbl) for lbl in pred_flat]

        cm = confusion_matrix(true_binary, pred_binary, labels=[1, 0])
        
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        cm_percent = cm_percent[::-1, :]
        
        x = [f"{et}", "Other"]
        y = ["Other", f"{et}"]
        
        fig = go.Figure(data=go.Heatmap(
            z=cm_percent,
            x=x,
            y=y,
            hoverongaps=False,
            colorscale='Blues',
            zmin=0,
            zmax=100,
            texttemplate="%{z:.2f}%"
        ))
        fig.update_layout(
            title=f"Confusion Matrix for {et}",
            xaxis=dict(title='Predicted Label'),
            yaxis=dict(title='True Label'),
            margin=dict(l=50, r=50, t=50, b=50),
            autosize=False,
            width=500,
            height=500,
        )
        
        all_figures[et] = fig
        all_cm_percent[et] = cm_percent
    for i, fig in all_figures:
        path = f'/data/horse/ws/irve354e-uniNer_test/energy_ner_llm/energy_ner_llm/src/roberta_uniner_inference/modeling/plots/{i}.png'
        fig.write_image(path)
        mlflow.log_artifact(path)
    return all_figures, all_cm_percent

def setup_mlflow_run(args, run_name):
    """Setup MLflow run and log parameters."""
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=run_name, log_system_metrics=True):
        mlflow.log_param("experiment_name", args.experiment_name)
        mlflow.log_param("run_name", run_name)
        mlflow.log_param("model_path", args.model_path)
        mlflow.log_param("dataset_name", args.dataset_name)
        mlflow.log_param("model_type", args.model_type)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("max_new_tokens", args.max_new_tokens)
        return mlflow.active_run()

def prepare_dataset(dataset_name):
    """Load and prepare dataset samples."""
    dataset = load_dataset(dataset_name)['test']
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
    return samples

def get_model_pipeline(model_type, model_path):
    """Initialize the appropriate model pipeline."""
    if model_type == "roberta":
        return pipeline(
            "ner",
            model=model_path,
            tokenizer=model_path,
            aggregation_strategy="simple"
        )
    elif model_type == "universalner":
        return pipeline(
            "text-generation",
            model=model_path,
            torch_dtype=torch.float16,
            device=0
        )
    raise ValueError(f"Unsupported model type: {model_type}")

def process_batch(batch, model_type, model_pipeline, args):
    """Process a batch of samples and return predictions."""
    inputs = [build_input(sample, model_type, args.entity_type) for sample in batch]
    
    if model_type == "roberta":
        outputs = model_pipeline(inputs)
        if outputs and isinstance(outputs[0], dict):
            outputs = [outputs]
    else:
        outputs = model_pipeline(inputs, max_new_tokens=args.max_new_tokens, return_full_text=False)
    
    batch_predictions = []
    batch_comparisons = []
    
    for sample, output in zip(batch, outputs):
        if model_type == "roberta":
            pred_labels = parse_roberta_output(sample, output)
        elif model_type == "universalner":
            pred_labels = parse_universalner_output(sample, output, args.entity_type)
        else:
            pred_labels = ["O"] * len(sample["tokens"])
            
        batch_predictions.append(pred_labels)
        batch_comparisons.append({
            "tokens": sample["tokens"],
            "true_labels": sample["true_labels"],
            "predicted_labels": pred_labels,
            "input": build_input(sample, model_type),
            "raw_output": output
        })
    
    return batch_predictions, batch_comparisons

def run_ner(args):
    """Run NER evaluation for either a Roberta-based or UniversalNER model."""
    run_name = (
        args.run_name
        if args.run_name != "default_run"
        else f"NER_{args.model_path.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    with mlflow.start_run(run_name=run_name, log_system_metrics=True):
        setup_mlflow_run(args, run_name)
        samples = prepare_dataset(args.dataset_name)
        model_pipeline = get_model_pipeline(args.model_type, args.model_path)

        effective_batch_size = len(samples) if args.use_full_batch else args.batch_size
        all_true_labels = [sample["true_labels"] for sample in samples]
        all_predicted_labels = []
        comparisons = []
        
        start_time = time.time()
        
        for i in range(0, len(samples), effective_batch_size):
            batch = samples[i : i + effective_batch_size]
            batch_predictions, batch_comparisons = process_batch(
                batch, args.model_type, model_pipeline, args
            )
            all_predicted_labels.extend(batch_predictions)
            comparisons.extend(batch_comparisons)

        total_time = time.time() - start_time
        mlflow.log_metric("total_processing_time", total_time)
        print(f"Total processing time: {total_time:.2f} seconds")

        results = evaluate_and_log(all_true_labels, all_predicted_labels, 
                                 comparisons, artifact_prefix=args.model_type)
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