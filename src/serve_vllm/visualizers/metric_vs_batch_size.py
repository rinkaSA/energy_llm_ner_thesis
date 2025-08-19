import glob
import os
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_PATTERN = os.path.join('/data/horse/ws/irve354e-energy_llm_ner/energy_ner_llm/qlora-ner/serve_vllm/INFERENCE_TWINER_1-128_awq_mistrall_full/telemetry', "telemetry_B_*.json")


def plot_metrics_vs_batch_size(
    file_pattern="telemetry_batch_*.json",
    metrics=None,
    cols=3,
    output_html="metrics_vs_batch_size.html"
):
    """
    Reads all JSON files matching `file_pattern`, extracts telemetry metrics,
    and creates a Plotly HTML with boxplots of each metric vs. batch_size.

    Parameters
    ----------
    file_pattern : str
        Glob pattern to locate your JSON files (e.g. "telemetry_batch_*.json").
    metrics : list of str, optional
        List of keys in the JSON to plot. If None, will infer all numeric columns.
    cols : int
        Number of subplot columns.
    output_html : str
        Path to save the generated HTML file.
    """
    dfs = []
    for filepath in glob.glob(file_pattern):
        batch_size = int(os.path.splitext(os.path.basename(filepath))[0].split("_")[-1])
        with open(filepath, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df["batch_size"] = batch_size
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    if metrics is None:
        metrics = [
            c for c in df.select_dtypes("number").columns
            if c != "batch_size"
        ]

    n = len(metrics)
    rows = 2
    cols = math.ceil(n / rows)

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=metrics,
        horizontal_spacing=0.02,
        vertical_spacing=0.1,
        shared_xaxes=False  # each subplot gets its own x‐axis
    )

    # 2. Add each box, give it a width, and force categorical x‐axis
    for i, metric in enumerate(metrics):
        r = (i // cols) + 1
        c = (i %  cols) + 1

        for b in sorted(df["batch_size"].unique()):
            subset = df[df["batch_size"] == b]
            fig.add_trace(
                go.Box(
                    y=subset[metric],
                    x=[str(b)] * len(subset),   # categorical x
                    name=str(b),
                    width=0.4,                  # nice fat boxes
                    boxmean="sd",
                    marker=dict(opacity=0.6),
                    showlegend=(i == 0)
                ),
                row=r, col=c
            )

        fig.update_xaxes(
            title_text="batch_size",
            type="category",
            row=r, col=c
        )
        fig.update_yaxes(title_text=metric, row=r, col=c)

    # 3. Final layout tweaks
    fig.update_layout(
        title="Telemetry Metrics vs. Batch Size",
        boxmode="group",
        margin=dict(l=40, r=40, t=80, b=40),
        height=400 * rows,
        width=500 * cols
    )


    fig.write_html(output_html)
    print(f"Saved plot to {output_html}.html")

plot_metrics_vs_batch_size(
    file_pattern=DEFAULT_PATTERN,
    metrics=[
        "latency_s",
        "prompt_tokens", 
        "generation_tokens",
        "total_tokens",

        "e2e_latency_mean", 
        "ttft_mean", # change to f
        "time_per_token_mean",

        "prefill_avg_s",
        "inference_avg_s",
        "decode_avg_s",

        "energy_j", 
        "joules_prefill",
        "joules_inference",
       "joules_decode",

      "J_prefill_per_prompt_token",
       "J_inf_per_gen_token",
        "J_per_total_token", 
        
        "J_total_per_prompt_token",
       "J_total_per_gen_token",
        "J_total_per_token",

        "avg_power_draw"
],
    cols=1,
    output_html="INFERENCE_TWINER_awq_mistrall_full_1_128.html"
)
