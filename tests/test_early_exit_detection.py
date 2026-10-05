import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

import early_exit_detection


class EarlyExitDetectionTests(unittest.TestCase):
    @patch("early_exit_detection.Path.resolve", return_value=Path("/virtual/project/early_exit_detection.py"))
    @patch("early_exit_detection.AutoModelForCausalLM.from_pretrained")
    @patch("early_exit_detection.AutoTokenizer.from_pretrained")
    @patch("early_exit_detection.load_from_disk")
    @patch("early_exit_detection.np.load")
    @patch("early_exit_detection.load_model_config")
    def test_main_evaluates_all_four_prompt_types(
        self,
        load_model_config,
        np_load,
        load_from_disk,
        load_tokenizer,
        load_model,
        resolve,
    ):
        load_model_config.return_value = ("test-model", 14, 2)

        rng = np.random.RandomState(42)
        n_prompts, n_layers, dim = 60, 5, 8
        fake_layer_data = {
            "benign_states": rng.randn(n_prompts, n_layers, dim),
            "malicious_states": rng.randn(n_prompts, n_layers, dim),
        }
        np_load.return_value = fake_layer_data

        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = "formatted"
        tokenizer.return_value = type("Inputs", (dict,), {
            "to": lambda self, dev: self,
        })({"input_ids": [[1, 2]]})
        load_tokenizer.return_value = tokenizer

        model = Mock()
        model.to.return_value = model
        model.model.layers = [Mock(), Mock(), Mock(), Mock(), Mock()]
        hidden_state = torch.ones((1, 2, dim), dtype=torch.float32)
        model.return_value = Mock(hidden_states=[hidden_state])
        load_model.return_value = model

        class SimpleDataset(dict):
            def __init__(self, col, data):
                super().__init__({col: data})
                self.column_names = [col]

            def __len__(self):
                return len(self[self.column_names[0]])

        ultrachat_mock = SimpleDataset("prompt", ["benign 1", "benign 2"])
        harmbench_mock = {"train": SimpleDataset("behavior", ["malicious 1", "malicious 2"])}
        gcg_mock = {"train": SimpleDataset("adv_suffix", ["suffix 1", "suffix 2"])}
        dan_mock = {"train": SimpleDataset("Prompt", ["dan 1", "dan 2"])}

        load_from_disk.side_effect = [
            ultrachat_mock,
            harmbench_mock,
            gcg_mock,
            dan_mock,
        ]

        with patch("builtins.print"):
            results = early_exit_detection.main()

        self.assertEqual(results["benign_total"], 2)
        self.assertEqual(results["malicious_total"], 2)
        self.assertEqual(results["gcg_total"], 2)
        self.assertEqual(results["roleplay_total"], 2)

        self.assertEqual(model.model.layers, model.model.layers[:2])
        self.assertEqual(load_from_disk.call_count, 4)
        load_model_config.assert_called_once_with(
            Path("/virtual/project/config/config.yml")
        )
        np_load.assert_called_once_with("./data/test-model_all_layers.npz")


if __name__ == "__main__":
    unittest.main()
