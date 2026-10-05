from pathlib import Path

import numpy as np
import torch
import pandas as pd
import random
import yaml
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_latent_vectors(model_id):
    """Load cached benign and malicious vectors and their prompts."""
    vector_path = f"./data/{model_id}_latent_vectors.npz"
    print(f"Loading cached vectors from {vector_path}...")
    with np.load(vector_path, allow_pickle=True) as data:
        return (
            data["X_benign"],
            data["X_malicious"],
            data["prompts_benign"],
            data["prompts_malicious"],
        )


def load_model_config(config_path):
    """Load the selected model and its latent-vector and AUC layer indices."""
    with Path(config_path).open() as config_file:
        config = yaml.safe_load(config_file)

    model_id = config["model"]["name"]
    model_config = config["models"].get(model_id)
    if model_config is None:
        raise ValueError(f"Model {model_id!r} is not defined in config.models")

    layer_idx = model_config["layer_idx"]
    auc_layer = model_config["auc_layer"]
    for layer_key, layer_value in (
        ("layer_idx", layer_idx),
        ("auc_layer", auc_layer),
    ):
        if not isinstance(layer_value, int) or layer_value < 0:
            raise ValueError(
                f"config.models[{model_id!r}].{layer_key} must be a non-negative integer"
            )
    return model_id, layer_idx, auc_layer


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

    print("Loading AdvBench from local disk...")
    try:
        advbench = pd.read_csv("./data/advbench.csv")
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


def load_stress_test_prompts(sample_limit=100):
    harmbench = load_from_disk("./data/harmbench")
    hb_columns = harmbench["train"].column_names
    hb_column = next(
        (column for column in ["Behavior", "behavior", "prompt"] if column in hb_columns),
        hb_columns[0],
    )
    base_attacks = harmbench["train"][hb_column]

    gcg_raw = load_from_disk("./data/adv_suffixes")
    gcg_dataset = gcg_raw[list(gcg_raw.keys())[0]]
    gcg_column = next(
        (
            column
            for column in ["adv_suffix", "adversarial_suffix", "suffix", "text", "prompt"]
            if column in gcg_dataset.column_names
        ),
        gcg_dataset.column_names[-1],
    )

    roleplay_raw = load_from_disk("./data/jailbreak_prompts")
    roleplay_dataset = roleplay_raw[list(roleplay_raw.keys())[0]]
    roleplay_column = next(
        (column for column in ["Prompt", "prompt", "text"] if column in roleplay_dataset.column_names),
        roleplay_dataset.column_names[0],
    )

    gcg_count = min(sample_limit, len(gcg_dataset), len(base_attacks))
    roleplay_count = min(sample_limit, len(roleplay_dataset), len(base_attacks))
    rng = random.Random(42)
    gcg_suffixes = rng.sample(gcg_dataset[gcg_column], gcg_count)
    roleplay_wrappers = rng.sample(roleplay_dataset[roleplay_column], roleplay_count)

    gcg_prompts = [
        f"{attack} {suffix}"
        for attack, suffix in zip(base_attacks[:gcg_count], gcg_suffixes)
    ]
    roleplay_prompts = [
        f"{wrapper}\n\nTask: {attack}"
        for attack, wrapper in zip(base_attacks[:roleplay_count], roleplay_wrappers)
    ]
    return gcg_prompts, roleplay_prompts