import torch
import numpy as np
import random
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk

def main():
    # ---------------------------------------------------------
    # 1. Reconstruct Layer 12 Harm Vector from Cache
    # ---------------------------------------------------------
    print("Loading cached layer data to calibrate the radar...")
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    data = np.load(f"./data/{model_id}_all_layers.npz")

    X_benign = data["benign_states"][:, 12, :]
    X_malicious = data["malicious_states"][:, 12, :]

    calib_size = 50
    mu_benign = np.mean(X_benign[:calib_size], axis=0)
    mu_malicious = np.mean(X_malicious[:calib_size], axis=0)

    harm_vector = mu_malicious - mu_benign
    harm_vector = harm_vector / np.linalg.norm(harm_vector)
    threshold = np.percentile(np.dot(X_benign, harm_vector), 99)
    print(f"Layer 12 Tripwire Threshold established at: {threshold:.4f}\n")

    # ---------------------------------------------------------
    # 2. Early-Exit Model Optimization
    # ---------------------------------------------------------
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Loading {model_id} and pruning architecture for early-exit...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.model.layers = model.model.layers[:13]
    print("Sliced model to 13 layers.\n")

    def score_prompt_early_exit(prompt):
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        raw_vector = outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()
        normalized_vector = raw_vector / np.linalg.norm(raw_vector)
        return np.dot(normalized_vector, harm_vector)

    # ---------------------------------------------------------
    # 3. Generate Stress Tests (Dynamic Column Resolution)
    # ---------------------------------------------------------
    print("Loading stress test datasets from local disk...")
    harmbench = load_from_disk("./data/harmbench")
    hb_columns = harmbench["train"].column_names
    hb_col = next((col for col in ["Behavior", "behavior", "prompt"] if col in hb_columns), hb_columns[0])
    base_attacks = harmbench["train"][hb_col][:100]

    # --- GCG Gibberish Suffixes ---
    gcg_raw = load_from_disk("./data/adv_suffixes")
    gcg_dataset = gcg_raw[list(gcg_raw.keys())[0]]

    print(f"GCG Columns detected: {gcg_dataset.column_names}")
    gcg_col = next((col for col in ["adv_suffix", "adversarial_suffix", "suffix", "text", "prompt"] if col in gcg_dataset.column_names), gcg_dataset.column_names[-1])

    # Cap sample size to prevent ValueError
    gcg_sample_size = min(100, len(gcg_dataset))
    random.seed(42)
    sampled_suffixes = random.sample(gcg_dataset[gcg_col], gcg_sample_size)
    gcg_prompts = [f"{attack} {suffix}" for attack, suffix in zip(base_attacks[:gcg_sample_size], sampled_suffixes)]

    # --- Roleplay / Developer Mode Wrappers ---
    dan_raw = load_from_disk("./data/jailbreak_prompts")
    dan_dataset = dan_raw[list(dan_raw.keys())[0]]

    print(f"DAN Columns detected: {dan_dataset.column_names}")
    dan_col = next((col for col in ["Prompt", "prompt", "text"] if col in dan_dataset.column_names), dan_dataset.column_names[0])

    # Cap sample size to prevent ValueError
    dan_sample_size = min(100, len(dan_dataset))
    sampled_jailbreaks = random.sample(dan_dataset[dan_col], dan_sample_size)
    roleplay_prompts = [f"{jailbreak}\n\nTask: {attack}" for attack, jailbreak in zip(base_attacks[:dan_sample_size], sampled_jailbreaks)]

    print(f"\nGenerated {len(gcg_prompts)} GCG attacks and {len(roleplay_prompts)} Roleplay attacks.")

    # ---------------------------------------------------------
    # 4. Evaluate Stress Tests
    # ---------------------------------------------------------
    print("\nTesting GCG Suffix Attacks...")
    gcg_caught = sum(
        score_prompt_early_exit(prompt) > threshold
        for prompt in tqdm(gcg_prompts)
    )

    print("\nTesting Roleplay/DAN Attacks...")
    roleplay_caught = sum(
        score_prompt_early_exit(prompt) > threshold
        for prompt in tqdm(roleplay_prompts)
    )

    print("\n" + "=" * 40)
    print("STRESS TEST RESULTS (LAYER 12 EARLY-EXIT)")
    print("=" * 40)
    print("Base Attacks Baseline (Expected): ~99%")
    print(f"GCG Suffixes Blocked:             {gcg_caught} / 100")
    print(f"Roleplay Wrappers Blocked:        {roleplay_caught} / 100")
    print("=" * 40)


if __name__ == "__main__":
    main()