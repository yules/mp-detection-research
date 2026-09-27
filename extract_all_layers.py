import torch
import numpy as np
import os
from tqdm import tqdm
from datasets import load_from_disk
from pathlib import Path
from lib.model_utils import (
    load_model_and_tokenizer,
    load_model_config,
    tokenize_prompt,
)


def get_all_layer_vectors(prompt, tokenizer, model, device):
    """Runs a single forward pass and extracts normalized vectors for ALL layers."""
    inputs = tokenize_prompt(prompt, tokenizer, device)

    with torch.no_grad():
        outputs = model(**inputs)

    layer_vectors = []
    for hidden_state in outputs.hidden_states:
        raw_vector = hidden_state[0, -1, :].float().cpu().numpy()
        normalized = raw_vector / np.linalg.norm(raw_vector)
        layer_vectors.append(normalized)

    return np.array(layer_vectors)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    config_path = Path(__file__).resolve().parent / "config" / "config.yml"
    model_id, _ = load_model_config(config_path)
    tokenizer, model = load_model_and_tokenizer(model_id, device)

    print("Loading subset of datasets for layer sweep...")

    try:
        harmbench = load_from_disk("./data/harmbench")
        hb_columns = harmbench["train"].column_names
        hb_col = next((col for col in ["Behavior", "behavior", "prompt"] if col in hb_columns), hb_columns[0])
        malicious_prompts = harmbench["train"][hb_col][:150]
    except Exception as e:
        print(f"Failed to load HarmBench: {e}")
        malicious_prompts = []

    try:
        ultrachat = load_from_disk("./data/ultrachat")
        benign_prompts = [msg[0]["content"] for msg in ultrachat["messages"][:150]]
    except Exception as e:
        print(f"Failed to load UltraChat: {e}")
        benign_prompts = []

    print("\nExtracting all layer states for Benign prompts...")
    all_benign_states = np.array([
        get_all_layer_vectors(prompt, tokenizer, model, device)
        for prompt in tqdm(benign_prompts)
    ])

    print("\nExtracting all layer states for Malicious prompts...")
    all_malicious_states = np.array([
        get_all_layer_vectors(prompt, tokenizer, model, device)
        for prompt in tqdm(malicious_prompts)
    ])

    os.makedirs("./data", exist_ok=True)
    np.savez(
        "./data/llama_all_layers.npz",
        benign_states=all_benign_states,
        malicious_states=all_malicious_states,
    )
    print("\nSuccessfully saved 3D layer states to ./data/llama_all_layers.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())