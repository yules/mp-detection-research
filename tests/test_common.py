import unittest
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pandas as pd
import torch
import yaml

import lib.common as common


class CommonTests(unittest.TestCase):
    @patch("lib.common.np.load")
    def test_load_latent_vectors_reads_expected_archive(self, load):
        benign = object()
        malicious = object()
        benign_prompts = object()
        malicious_prompts = object()
        archive = {
            "X_benign": benign,
            "X_malicious": malicious,
            "prompts_benign": benign_prompts,
            "prompts_malicious": malicious_prompts,
        }
        load.return_value.__enter__.return_value = archive

        vectors = common.load_latent_vectors("test-model")

        self.assertEqual(
            vectors, (benign, malicious, benign_prompts, malicious_prompts)
        )
        load.assert_called_once_with(
            "./data/test-model_latent_vectors.npz", allow_pickle=True
        )

    def test_load_model_config_returns_selected_model_and_layer(self):
        config = {
            "model": {"name": "test-model"},
            "models": {
                "test-model": {"layer_idx": 3, "auc_layer": 5},
            },
        }
        config_path = Path("/virtual/config.yml")
        with patch(
            "pathlib.Path.open", mock_open(read_data=yaml.safe_dump(config))
        ):
            result = common.load_model_config(config_path)

        self.assertEqual(result, ("test-model", 3, 5))

    def test_load_model_config_rejects_invalid_layer(self):
        config = {
            "model": {"name": "test-model"},
            "models": {
                "test-model": {"layer_idx": -1, "auc_layer": 5},
            },
        }
        with patch(
            "pathlib.Path.open", mock_open(read_data=yaml.safe_dump(config))
        ):
            with self.assertRaisesRegex(ValueError, "non-negative integer"):
                common.load_model_config(Path("/virtual/config.yml"))

    def test_load_model_config_rejects_invalid_auc_layer(self):
        config = {
            "model": {"name": "test-model"},
            "models": {
                "test-model": {"layer_idx": 3, "auc_layer": -1},
            },
        }
        with patch(
            "pathlib.Path.open", mock_open(read_data=yaml.safe_dump(config))
        ):
            with self.assertRaisesRegex(ValueError, "non-negative integer"):
                common.load_model_config(Path("/virtual/config.yml"))

    @patch("lib.common.AutoModelForCausalLM.from_pretrained")
    @patch("lib.common.AutoTokenizer.from_pretrained")
    def test_load_model_and_tokenizer_configures_padding_and_device(
        self, from_pretrained_tok, from_pretrained_model
    ):
        tokenizer = Mock(pad_token=None, eos_token="<eos>")
        from_pretrained_tok.return_value = tokenizer

        model = Mock()
        from_pretrained_model.return_value = model
        device = torch.device("cpu")

        actual_tokenizer, actual_model = common.load_model_and_tokenizer(
            "test-model", device
        )

        self.assertIs(actual_tokenizer, tokenizer)
        self.assertEqual(tokenizer.pad_token, "<eos>")
        self.assertIs(actual_model, model.to.return_value)
        from_pretrained_tok.assert_called_once_with(
            "test-model", padding_side="left"
        )
        from_pretrained_model.assert_called_once_with(
            "test-model",
            output_hidden_states=True,
            torch_dtype=torch.bfloat16,
        )
        model.to.assert_called_once_with(device)

    def test_tokenize_prompt_formats_and_moves_inputs_to_device(self):
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = "formatted prompt"
        inputs = Mock()
        tokenizer.return_value = inputs
        device = torch.device("cpu")

        result = common.tokenize_prompt("hello", tokenizer, device)

        self.assertIs(result, inputs.to.return_value)
        tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "hello"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        tokenizer.assert_called_once_with(
            "formatted prompt",
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        inputs.to.assert_called_once_with(device)

    def test_tokenize_prompt_falls_back_when_no_chat_template(self):
        tokenizer = Mock()
        tokenizer.chat_template = None
        inputs = Mock()
        tokenizer.return_value = inputs
        device = torch.device("cpu")

        result = common.tokenize_prompt("hello", tokenizer, device)

        self.assertIs(result, inputs.to.return_value)
        tokenizer.apply_chat_template.assert_not_called()
        tokenizer.assert_called_once_with(
            "hello",
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        inputs.to.assert_called_once_with(device)

    def test_tokenize_prompt_falls_back_when_apply_chat_template_raises(self):
        tokenizer = Mock()
        tokenizer.apply_chat_template.side_effect = ValueError("No chat template")
        inputs = Mock()
        tokenizer.return_value = inputs
        device = torch.device("cpu")

        result = common.tokenize_prompt("hello", tokenizer, device)

        self.assertIs(result, inputs.to.return_value)
        tokenizer.assert_called_once_with(
            "hello",
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        inputs.to.assert_called_once_with(device)

    @patch("lib.common.pd.read_csv")
    @patch("lib.common.load_from_disk")
    def test_load_prompt_sets_extracts_and_limits_all_sources(
        self, load_from_disk, read_csv
    ):
        class HarmBenchTrain(dict):
            column_names = ["behavior"]

        load_from_disk.side_effect = [
            {"train": HarmBenchTrain(behavior=["hb-1", "hb-2", "hb-3"])},
            {"messages": [[{"content": "benign-1"}], [{"content": "benign-2"}]]},
        ]
        read_csv.return_value["goal"].tolist.return_value = ["adv-1", "adv-2"]

        with patch("builtins.print"):
            benign_prompts, malicious_prompts = common.load_prompt_sets(
                harmbench_limit=2, ultrachat_limit=1, malicious_limit=3
            )

        self.assertEqual(benign_prompts, ["benign-1"])
        self.assertEqual(malicious_prompts, ["hb-1", "hb-2", "adv-1"])
        self.assertEqual(
            load_from_disk.call_args_list,
            [
                unittest.mock.call("./data/harmbench"),
                unittest.mock.call("./data/ultrachat"),
            ],
        )

    @patch("lib.common.load_from_disk", side_effect=RuntimeError("missing"))
    @patch("lib.common.pd.read_csv")
    def test_load_prompt_sets_requires_benign_when_requested(
        self, read_csv, load_from_disk
    ):
        read_csv.return_value["goal"].tolist.return_value = []
        with patch("builtins.print"):
            with self.assertRaisesRegex(RuntimeError, "missing"):
                common.load_prompt_sets(require_benign=True)

    @patch("lib.common.load_from_disk")
    def test_load_stress_test_prompts_constructs_limited_evaluation_prompts(
        self, load_from_disk
    ):
        class Dataset(dict):
            def __init__(self, column, values):
                super().__init__({column: values})
                self.column_names = [column]

            def __len__(self):
                return len(self[self.column_names[0]])

        class HarmBenchTrain(dict):
            column_names = ["Behavior"]

        load_from_disk.side_effect = [
            {"train": HarmBenchTrain(Behavior=["attack-1", "attack-2"])},
            {"train": Dataset("suffix", ["suffix-1", "suffix-2"])},
            {"train": Dataset("Prompt", ["wrapper-1", "wrapper-2"])},
        ]

        gcg_prompts, roleplay_prompts = common.load_stress_test_prompts(
            sample_limit=1
        )

        self.assertEqual(len(gcg_prompts), 1)
        self.assertEqual(len(roleplay_prompts), 1)
        self.assertTrue(gcg_prompts[0].startswith("attack-1 "))
        self.assertEqual(
            roleplay_prompts[0], "wrapper-1\n\nTask: attack-1"
        )
        self.assertEqual(
            load_from_disk.call_args_list,
            [
                unittest.mock.call("./data/harmbench"),
                unittest.mock.call("./data/adv_suffixes"),
                unittest.mock.call("./data/jailbreak_prompts"),
            ],
        )

    def test_to_dataset_extracts_from_dict_and_dataset_dict(self):
        class Simple(dict):
            column_names = ["col1"]

        # Case 1: Dict with train split
        data1 = {"train": Simple(col1=[1, 2])}
        self.assertEqual(common.to_dataset(data1), data1["train"])

        # Case 2: Dict without train split
        data2 = {"custom": Simple(col1=[3, 4])}
        self.assertEqual(common.to_dataset(data2), data2["custom"])

        # Case 3: Already dataset
        raw = Simple(col1=[5, 6])
        self.assertEqual(common.to_dataset(raw), raw)

    def test_get_device_returns_torch_device(self):
        device = common.get_device()
        self.assertIsInstance(device, torch.device)

    def test_evaluate_repe_status_outcomes(self):
        cases = [
            (
                {"Ground_Truth": "Malicious", "Prediction": "Malicious"},
                "True Anomaly (Detected Attack)",
            ),
            (
                {"Ground_Truth": "Malicious", "Prediction": "Benign"},
                "Missed Attack (False Negative)",
            ),
            (
                {"Ground_Truth": "Benign", "Prediction": "Malicious"},
                "False Alarm (False Positive)",
            ),
            (
                {"Ground_Truth": "Benign", "Prediction": "Benign"},
                "Benign (Passed)",
            ),
        ]
        for row, expected in cases:
            self.assertEqual(common.evaluate_repe_status(row), expected)

    @patch("lib.common.pd.DataFrame.to_csv", autospec=True)
    def test_export_results_to_csv_filters_and_sorts(self, mock_to_csv):
        df = pd.DataFrame({
            "Result_Status": [
                "Benign (Passed)",
                "False Alarm (False Positive)",
                "True Anomaly (Detected Attack)",
                "Missed Attack (False Negative)",
            ],
            "RepE_Harm_Score": [0.1, 0.9, 0.8, 0.4],
            "Prompt_Text": ["p1", "p2", "p3", "p4"],
        })

        with patch("builtins.print"):
            csv_path = common.export_results_to_csv(df, "test_results")

        self.assertTrue(csv_path.startswith("./data/test_results_"))
        self.assertTrue(csv_path.endswith(".csv"))
        mock_to_csv.assert_called_once()
        exported_df = mock_to_csv.call_args.args[0]
        self.assertEqual(len(exported_df), 2)
        self.assertEqual(
            exported_df["Result_Status"].tolist(),
            ["False Alarm (False Positive)", "Missed Attack (False Negative)"],
        )
        self.assertEqual(exported_df["RepE_Harm_Score"].tolist(), [0.9, 0.4])


if __name__ == "__main__":
    unittest.main()