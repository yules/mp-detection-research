import os
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from lib.model_utils import (
    load_model_and_tokenizer,
    load_model_config,
    load_prompt_sets,
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
    model_id, layer_idx, _ = load_model_config(config_path)

    print(f"Loading {model_id} onto GPU...")
    tokenizer, model = load_model_and_tokenizer(model_id, device)

    print("\nLoading datasets...")
    try:
        benign_prompts, malicious_prompts = load_prompt_sets(require_benign=True)
    except Exception as e:
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
    output_path = f"./data/{model_id}_latent_vectors.npz"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(
        output_path,
        X_benign=X_benign,
        X_malicious=X_malicious,
        prompts_benign=np.array(benign_prompts),
        prompts_malicious=np.array(malicious_prompts)
    )
    print(f"\nSuccessfully saved raw vectors to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())