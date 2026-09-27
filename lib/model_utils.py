from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


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