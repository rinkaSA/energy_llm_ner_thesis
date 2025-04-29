from datasets import load_dataset
import mlflow
from openai import OpenAI
import time 
import json
import evaluate
import  numpy as np
import matplotlib.pyplot as plt
import ast
from dotenv import load_dotenv
import re
import plotly.graph_objects as go
import plotly.express as px


load_dotenv()
try:
    with open("/data/horse/ws/irve354e-uniNer_test/energy_ner_llm/energy_ner_llm/src/mod_distil_by_cot/my_key_scads") as keyfile:
       my_api_key = keyfile.readline().strip()
except FileNotFoundError:
    print("file 'my_key_scads' was not found")
    exit(1)


prompt_CoT_CONLL = '''
You are an expert of natural language processing annotation, given a sentence, you are going to identify and classify each named entity according to its type: LOC (Location), MISC (Miscellaneous), ORG (Organization), or PER (Person). Show your reasoning process in steps before providing the results in a structured format.

NER types:
1. LOC (Location): Identifies geographical entities such as countries, cities, rivers, and mountains.
2. MISC (Miscellaneous): Categorizes entities that don't clearly fall into the other standard types like organizations, persons, or locations.
3. ORG (Organization): Marks specific organizations, including companies, governmental bodies, and non-governmental organizations.
4. PER (Person): Used for the names of individuals, identifying people in the text.

Follow these steps to annotate the sentence. 
Step 1.#### Read the sentence and understand its context.
Step 2.#### Identify potential named entities within the sentence.
Step 3.#### Determine the type of each entity (LOC, MISC, ORG, PER) based on the context.
Step 4.#### Justify the classification of each entity with reasoning. 

Use the following format:
Step 1.#### <step 1 reasoning>
Step 2.#### <step 2 reasoning>
Step 3.#### <step 3 reasoning>
Step 4.#### <final output>
Make sure to include #### to separate every step.

Sentence: 'In Houston , Orlando Miller 's two-run homer with one out in the bottom of the ninth off Todd Stottlemyre gave the Houston Astros a 3-1 win over the St. Louis Cardinals and left the teams in a virtual tie for the lead in the NL Central division .'
Step 1.#### The sentence narrates a significant moment in a baseball game, where Orlando Miller hits a two-run homer off Todd Stottlemyre, leading to a win for the Houston Astros against the St. Louis Cardinals, impacting their position in the NL Central division.
Step 2.#### The entities identified are Houston, Orlando Miller, Todd Stottlemyre, Houston Astros, St. Louis Cardinals, and the NL Central division.
Step 3.#### Houston is classified as a location (LOC), as it refers to a city. Orlando Miller and Todd Stottlemyre are classified as persons (PER), as they are individual names. Houston Astros and St. Louis Cardinals are classified as organizations (ORG), as they are names of baseball teams. The NL Central division is classified as miscellaneous (MISC), as it refers to a specific division within a sports league rather than a standard location, person, or organization.
Step 4.#### {{'LOC': ['Houston'], 'PER': ['Orlando Miller', 'Todd Stottlemyre'], 'ORG': ['Houston Astros', 'St. Louis Cardinals', 'NL Central division']}}

Sentence: 'Prime Minister Benjamin Netanyahu 's government , which took office in June , has said it will not allow the Authority , set up under a 1993 interim peace deal to control parts of the Gaza Strip and West Bank , to operate in Jerusalem .'
Step 1.#### The sentence describes the stance of Prime Minister Benjamin Netanyahu's government on the operational scope of the Authority in Jerusalem, set up under a 1993 interim peace deal, and involving geographical regions like the Gaza Strip and West Bank.
Step 2.#### The identified entities are Benjamin Netanyahu, Authority, Gaza Strip, West Bank, and Jerusalem.
Step 3.#### Benjamin Netanyahu is classified as a person (PER) since he is an individual. Authority（ORG） is an organizational entity as it refers to an administrative or political body. Gaza Strip（LOC）, West Bank（LOC）, and Jerusalem（LOC） are classified as locations since they refer to geographical areas.
Step 4.#### {{'LOC': ['Gaza Strip', 'West Bank', 'Jerusalem'], 'PER': ['Benjamin Netanyahu'], 'ORG': ['Authority']}}

Sentence: 'Brazilian Planning Minister Antonio Kandir will submit to a draft copy of the 1997 federal budget to Congress on Thursday , a ministry spokeswoman said .'
Step 1.#### The sentence describes an action by Antonio Kandir, the Brazilian Planning Minister, who is planning to submit a draft of the 1997 federal budget to Congress, as stated by a ministry spokeswoman.
Step 2.#### The entities identified are Brazilian (as an adjective related to Antonio Kandir), Antonio Kandir, and Congress.
Step 3.#### The term 'Brazilian' is associated with Antonio Kandir and is classified as miscellaneous (MISC), as it describes a nationality. Antonio Kandir is classified as a person (PER), as it is an individual's name. Congress is classified as an organization (ORG), as it refers to a governmental legislative body.
Step 4.#### {{'MISC': ['Brazilian'], 'PER': ['Antonio Kandir'], 'ORG': ['Congress']}}

Sentence: '{}'
'''

prompt_standard_CONLL = '''
You are an expert of natural language processing annotation, given a sentence, you are going to identify and classify each named entity according to its type: LOC (Location), MISC (Miscellaneous), ORG (Organization), or PER (Person).

NER types:
1. LOC (Location): Identifies geographical entities such as countries, cities, rivers, and mountains.
2. MISC (Miscellaneous): Categorizes entities that don't clearly fall into the other standard types like organizations, persons, or locations.
3. ORG (Organization): Marks specific organizations, including companies, governmental bodies, and non-governmental organizations.
4. PER (Person): Used for the names of individuals, identifying people in the text.

Sentence: 'In Houston , Orlando Miller 's two-run homer with one out in the bottom of the ninth off Todd Stottlemyre gave the Houston Astros a 3-1 win over the St. Louis Cardinals and left the teams in a virtual tie for the lead in the NL Central division .'
Result: {{'LOC': ['Houston'], 'PER': ['Orlando Miller', 'Todd Stottlemyre'], 'ORG': ['Houston Astros', 'St. Louis Cardinals', 'NL Central division']}}

Sentence: 'Prime Minister Benjamin Netanyahu 's government , which took office in June , has said it will not allow the Authority , set up under a 1993 interim peace deal to control parts of the Gaza Strip and West Bank , to operate in Jerusalem .'
Result: {{'LOC': ['Gaza Strip', 'West Bank', 'Jerusalem'], 'PER': ['Benjamin Netanyahu'], 'ORG': ['Authority']}}

Sentence: 'Brazilian Planning Minister Antonio Kandir will submit to a draft copy of the 1997 federal budget to Congress on Thursday , a ministry spokeswoman said .'
Result: {{'MISC': ['Brazilian'], 'PER': ['Antonio Kandir'], 'ORG': ['Congress']}}

Sentence: '{}'
'''


def np_encoder(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def extract_json_with_split(generated_text, marker="Step 4:####"):
    if marker in generated_text:
        json_part = generated_text.split(marker)[-1].strip()
        try:
            result = json.loads(json_part)
            if isinstance(result, dict):
                return result
        except Exception:
            try:
                result = ast.literal_eval(json_part)
                if isinstance(result, dict):
                    return result
            except Exception:
                pass
    return None

def extract_last_json_block(generated_text):
    matches = re.findall(r"(\{.*?\})", generated_text, re.DOTALL)
    if matches:
        for candidate in reversed(matches):
            candidate = candidate.strip()
            try:
                result = json.loads(candidate)
                if isinstance(result, dict):
                    return result
            except Exception:
                try:
                    result = ast.literal_eval(candidate)
                    if isinstance(result, dict):
                        return result
                except Exception:
                    continue
    return {}

def parse_output(sample, output):
    """
    The output is expected to be a multi-step reasoning string where the final step contains
    a JSON or Python dictionary mapping entity types to lists of entity phrases.
    
    This function first attempts to use a simple split based on a marker ("Step 4:####").
    If that fails to yield a valid dictionary, it falls back to a regex-based extraction.
    
    It returns token-level BIO labels computed from the parsed entity mapping and the full reasoning.
    """
    generated_text = output[0].get("generated_text", "")
    reasoning = generated_text.strip()
    
    predicted_entities_by_type = extract_json_with_split(generated_text, marker="Step 4:####")
    
    if predicted_entities_by_type is None:
        predicted_entities_by_type = extract_last_json_block(generated_text)
    
    tokens = sample["tokens"]
    labels = ["O"] * len(tokens)
    
    for entity_type, entities in predicted_entities_by_type.items():
        for entity in entities:
            if not isinstance(entity, str):
                continue
            entity_tokens = entity.split()
            n = len(entity_tokens)
            for i in range(len(tokens) - n + 1):
                if tokens[i:i+n] == entity_tokens:
                    labels[i] = f"B-{entity_type}"
                    for j in range(1, n):
                        labels[i+j] = f"I-{entity_type}"
    return labels, reasoning



def evaluate_and_log(all_true_labels, all_predicted_labels, comparisons, artifact_prefix):
    """
    Compute seqeval metrics and log overall as well as per-entity type scores.
    Detailed results (including comparisons) are saved as a JSON artifact.
    """
    seqeval_metric = evaluate.load("seqeval")
    results = seqeval_metric.compute(predictions=all_predicted_labels, references=all_true_labels)


    overall_metrics = {}

    for key, value in results.items():
        if key.startswith("overall_"):
            mlflow.log_metric(key, value)
            overall_metrics[key] = value 
    entity_metrics = {}
    for entity, scores in results.items():
        if isinstance(scores, dict):
            entity_metrics[entity] = scores
            for metric, score in scores.items():
                mlflow.log_metric(f"{entity}_{metric}", score)
    
    detailed_results = {
        "comparisons": comparisons,
        "evaluation": results
    }
    detailed_json = f"{artifact_prefix}_detailed_results.json"
    with open(detailed_json, "w") as f:
        json.dump(detailed_results, f, indent=2, default=np_encoder)
    mlflow.log_artifact(detailed_json)

    plot_overall_metrics(overall_metrics)
    plot_per_entity_metrics(entity_metrics)
    
    return results

def plot_overall_metrics(overall_metrics):
    metrics = list(overall_metrics.keys())
    values = list(overall_metrics.values())

    fig = go.Figure(data=[
        go.Bar(x=metrics, y=values, text=[f"{v:.2f}" for v in values], textposition='auto')
    ])
    
    fig.update_layout(
        title="Overall Metrics",
        xaxis_title="Metric",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        template="plotly_white"
    )

    output_file = "energy_ner_llm/energy_ner_llm/src/mod_distil_by_cot/plots/overall_metrics.png"
    fig.write_image(output_file)
    mlflow.log_artifact(output_file)

def plot_per_entity_metrics(entity_metrics):
    entities = []
    metrics = []
    values = []

    for entity, scores in entity_metrics.items():
        for metric, value in scores.items():
            entities.append(entity)
            metrics.append(metric)
            values.append(value)

    df = {
        "Entity": entities,
        "Metric": metrics,
        "Value": values
    }

    fig = px.bar(df, x="Entity", y="Value", color="Metric", barmode="group",
                 title="Per-Entity Metrics",
                 labels={"Value": "Score", "Entity": "Named Entity Type"},
                 template="plotly_white")

    fig.update_yaxes(range=[0, 1])
    output_file = "energy_ner_llm/energy_ner_llm/src/mod_distil_by_cot/plots/operentity_metrics.png"
    fig.write_image(output_file)
    mlflow.log_artifact(output_file)

def annotate_sample(joined_text):

    prompt = prompt_CoT_CONLL.format(joined_text)
    messages = [{'role': 'user', 'content': prompt}]
    response = client.chat.completions.create(model=model_name, messages=messages)
    generated_text = response.choices[0].message.content.strip()
    return [{"generated_text": generated_text}]


client = OpenAI(base_url="https://llm.scads.ai/v1",api_key=my_api_key)
model_name = 'meta-llama/Llama-3.3-70B-Instruct'


dataset = load_dataset("conll2003")["train"]
selected_dataset = dataset.shuffle(seed=12).select(range(100))

all_true_labels = []
all_predicted_labels = []
comparisons = []

total_llm_time = 0.0

label_names = dataset.features["ner_tags"].feature.names


mlflow.set_experiment("ConLL2003_Annotation_Experiment")
with mlflow.start_run(run_name='cot_annotation', log_system_metrics=True) as run:
    mlflow.log_param("model", model_name)
    mlflow.log_param("dataset", "conll2003-train (100 random samples)")
    mlflow.log_param("seed", 12)
    
    overall_start_time = time.time()
    i = 0
    for sample in selected_dataset:
        tokens = sample["tokens"]
        joined_text = " ".join(tokens)
        true_labels = [label_names[tag] for tag in sample["ner_tags"]]
        
        start_llm = time.time()
        output = annotate_sample(joined_text)
        llm_time = time.time() - start_llm
        total_llm_time += llm_time
        print(i)
        
        predicted_labels, reasoning = parse_output(sample, output)
        
        all_true_labels.append(true_labels)
        all_predicted_labels.append(predicted_labels)
        
        comparisons.append({
            "text": joined_text,
            "true_labels": true_labels,
            "predicted_labels": predicted_labels,
            "raw_output": output,
            "reasoning": reasoning,
            "llm_time": llm_time
        })
        i+=1
    
    overall_total_time = time.time() - overall_start_time
    mlflow.log_metric("total_llm_time_seconds", total_llm_time)
    mlflow.log_metric("average_llm_time_per_sample_seconds", total_llm_time / len(selected_dataset))
    
    comparisons_file = "comparisons.json"
    with open(comparisons_file, "w") as f:
        json.dump(comparisons, f, indent=2, default=np_encoder)
    mlflow.log_artifact(comparisons_file)
    
    artifact_prefix = "conll_train_1000"
    eval_results = evaluate_and_log(all_true_labels, all_predicted_labels, comparisons, artifact_prefix)
    
    mlflow.log_metric("total_samples", len(selected_dataset))
    
    print(f"MLflow run completed. Run ID: {run.info.run_id}")
    print(f"Total LLM call time for annotation: {total_llm_time:.2f} seconds")
    print(f"Overall processing time (including parsing): {overall_total_time:.2f} seconds")
    print("Evaluation results:", eval_results)
