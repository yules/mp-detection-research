from pathlib import Path

import torch
import pandas as pd
import yaml
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer


ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_behaviors.csv"
)


def load_model_config(config_path):
    """Load the selected model and its latent-vector layer index."""
    with Path(config_path).open() as config_file:
        config = yaml.safe_load(config_file)

    model_id = config["model"]["name"]
    model_config = config["models"].get(model_id)
    if model_config is None:
        raise ValueError(f"Model {model_id!r} is not defined in config.models")

    layer_idx = model_config["layer_idx"]
    if not isinstance(layer_idx, int) or layer_idx < 0:
        raise ValueError(
            f"config.models[{model_id!r}].layer_idx must be a non-negative integer"
        )
    return model_id, layer_idx


def load_model_and_tokenizer(model_id, device):
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        output_hidden_states=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    return tokenizer, model


def tokenize_prompt(prompt, tokenizer, device, max_length=1024):
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)


def load_prompt_sets(
    harmbench_limit=200,
    ultrachat_limit=2000,
    malicious_limit=400,
    require_benign=False,
):
    try:
        harmbench = load_from_disk("./data/harmbench")
        hb_columns = harmbench["train"].column_names
        hb_column = next(
            (column for column in ["Behavior", "behavior", "prompt"] if column in hb_columns),
            hb_columns[0],
        )
        harmbench_prompts = harmbench["train"][hb_column][:harmbench_limit]
    except Exception as error:
        print(f"Failed to load HarmBench: {error}")
        harmbench_prompts = []

    print("Downloading AdvBench to supplement malicious baseline...")
    try:
        advbench = pd.read_csv(ADVBENCH_URL)
        advbench_prompts = advbench["goal"].tolist()
    except Exception as error:
        print(f"Failed to load AdvBench: {error}")
        advbench_prompts = []

    try:
        ultrachat = load_from_disk("./data/ultrachat")
        benign_prompts = [
            message[0]["content"]
            for message in ultrachat["messages"][:ultrachat_limit]
        ]
    except Exception as error:
        print(f"Failed to load UltraChat: {error}")
        if require_benign:
            raise
        benign_prompts = []

    malicious_prompts = (
        list(harmbench_prompts) + advbench_prompts
    )[:malicious_limit]
    return benign_prompts, malicious_prompts