import unittest
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import torch
import yaml

import lib.model_utils as model_utils


class ModelUtilsTests(unittest.TestCase):
    def test_load_model_config_returns_selected_model_and_layer(self):
        config = {
            "model": {"name": "test-model"},
            "models": {"test-model": {"layer_idx": 3}},
        }
        config_path = Path("/virtual/config.yml")
        with patch(
            "pathlib.Path.open", mock_open(read_data=yaml.safe_dump(config))
        ):
            result = model_utils.load_model_config(config_path)

        self.assertEqual(result, ("test-model", 3))

    def test_load_model_config_rejects_invalid_layer(self):
        config = {
            "model": {"name": "test-model"},
            "models": {"test-model": {"layer_idx": -1}},
        }
        config_path = Path("/virtual/config.yml")
        with patch(
            "pathlib.Path.open", mock_open(read_data=yaml.safe_dump(config))
        ):
            with self.assertRaisesRegex(ValueError, "non-negative integer"):
                model_utils.load_model_config(config_path)

    @patch("lib.model_utils.AutoModelForCausalLM.from_pretrained")
    @patch("lib.model_utils.AutoTokenizer.from_pretrained")
    def test_load_model_and_tokenizer_configures_padding_and_model(
        self, load_tokenizer, load_model
    ):
        tokenizer = Mock(pad_token=None, eos_token="eos")
        model = Mock()
        model.to.return_value = model
        load_tokenizer.return_value = tokenizer
        load_model.return_value = model
        device = torch.device("cpu")

        actual_tokenizer, actual_model = model_utils.load_model_and_tokenizer(
            "test-model", device
        )

        self.assertIs(actual_tokenizer, tokenizer)
        self.assertIs(actual_model, model)
        self.assertEqual(tokenizer.pad_token, "eos")
        load_tokenizer.assert_called_once_with("test-model", padding_side="left")
        load_model.assert_called_once_with(
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

        result = model_utils.tokenize_prompt("hello", tokenizer, device)

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

    @patch("lib.model_utils.pd.read_csv")
    @patch("lib.model_utils.load_from_disk")
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
            benign_prompts, malicious_prompts = model_utils.load_prompt_sets(
                harmbench_limit=2,
                ultrachat_limit=1,
                malicious_limit=3,
            )

        self.assertEqual(benign_prompts, ["benign-1"])
        self.assertEqual(malicious_prompts, ["hb-1", "hb-2", "adv-1"])
        load_from_disk.assert_any_call("./data/harmbench")
        load_from_disk.assert_any_call("./data/ultrachat")
        read_csv.assert_called_once_with("./data/advbench.csv")

    @patch("lib.model_utils.load_from_disk", side_effect=RuntimeError("missing"))
    @patch("lib.model_utils.pd.read_csv")
    def test_load_prompt_sets_requires_benign_when_requested(
        self, read_csv, load_from_disk
    ):
        read_csv.return_value["goal"].tolist.return_value = []
        with patch("builtins.print"):
            with self.assertRaisesRegex(RuntimeError, "missing"):
                model_utils.load_prompt_sets(require_benign=True)

    @patch("lib.model_utils.load_from_disk")
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

        gcg_prompts, roleplay_prompts = model_utils.load_stress_test_prompts(
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


if __name__ == "__main__":
    unittest.main()