import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datasets import load_from_disk
from tqdm import tqdm
from lib.model_utils import (
    load_model_and_tokenizer,
    load_model_config,
    tokenize_prompt,
)


def get_latent_vector_sequential(prompt, tokenizer, model, device, layer_idx):
    inputs = tokenize_prompt(prompt, tokenizer, device)

    with torch.no_grad():
        outputs = model(**inputs)
        
    hidden_states = outputs.hidden_states[layer_idx]
    
    raw_vector = hidden_states[0, -1, :].float().cpu().numpy()
    return raw_vector / np.linalg.norm(raw_vector)


def main():
    # ---------------------------------------------------------
    # 1. Hardware & Model Setup
    # ---------------------------------------------------------
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Hardware backend initialized: {device}")

    config_path = Path(__file__).resolve().parent / "config" / "config.yml"
    model_id, layer_idx = load_model_config(config_path)

    print(f"Loading {model_id} onto GPU...")
    tokenizer, model = load_model_and_tokenizer(model_id, device)

    # ---------------------------------------------------------
    # 2. Dataset Setup
    # ---------------------------------------------------------
    print("\nLoading datasets...")
    try:
        harmbench = load_from_disk("./data/harmbench")
        hb_columns = harmbench["train"].column_names
        hb_col = next((col for col in ["Behavior", "behavior", "prompt"] if col in hb_columns), hb_columns[0])
        hb_prompts = harmbench["train"][hb_col][:200]
    except Exception as e:
        print(f"Failed to load HarmBench: {e}")
        hb_prompts = []

    print("Downloading AdvBench to supplement malicious baseline...")
    advbench_url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    try:
        advbench_df = pd.read_csv(advbench_url)
        advbench_prompts = advbench_df['goal'].tolist()
    except Exception as e:
        print(f"Failed to load AdvBench: {e}")
        advbench_prompts = []

    malicious_prompts = (list(hb_prompts) + advbench_prompts)[:400]

    try:
        ultrachat = load_from_disk("./data/ultrachat")
        benign_prompts = [msg[0]["content"] for msg in ultrachat["messages"][:2000]]
    except Exception as e:
        print(f"Failed to load UltraChat: {e}")
        return 1

    print(f"Loaded {len(benign_prompts)} benign and {len(malicious_prompts)} malicious prompts.")

    # ---------------------------------------------------------
    # 3. Sequential Vector Extraction
    # ---------------------------------------------------------
    print("\nExtracting Benign latent vectors (Sequential - No Swap)...")
    X_benign = np.array([
        get_latent_vector_sequential(p, tokenizer, model, device, layer_idx)
        for p in tqdm(benign_prompts)
    ])

    print("\nExtracting Malicious latent vectors (Sequential - No Swap)...")
    X_malicious = np.array([
        get_latent_vector_sequential(p, tokenizer, model, device, layer_idx)
        for p in tqdm(malicious_prompts)
    ])

    # ---------------------------------------------------------
    # 4. Serialize to Disk
    # ---------------------------------------------------------
    print("\nStructuring data for export...")
    os.makedirs("./data", exist_ok=True)
    np.savez(
        "./data/latent_vectors.npz",
        X_benign=X_benign,
        X_malicious=X_malicious,
        prompts_benign=np.array(benign_prompts),
        prompts_malicious=np.array(malicious_prompts)
    )
    print("\nSuccessfully saved raw vectors to ./data/latent_vectors.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())