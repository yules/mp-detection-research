import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
import yaml

import lib.model_utils as model_utils


class ModelUtilsTests(unittest.TestCase):
    def test_load_model_config_returns_selected_model_and_layer(self):
        config = {
            "model": {"name": "test-model"},
            "models": {"test-model": {"layer_idx": 3}},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yml"
            config_path.write_text(yaml.safe_dump(config))

            result = model_utils.load_model_config(config_path)

        self.assertEqual(result, ("test-model", 3))

    def test_load_model_config_rejects_invalid_layer(self):
        config = {
            "model": {"name": "test-model"},
            "models": {"test-model": {"layer_idx": -1}},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yml"
            config_path.write_text(yaml.safe_dump(config))

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


if __name__ == "__main__":
    unittest.main()