import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
import yaml

import extract_vectors


class ExtractVectorsTests(unittest.TestCase):
    def test_load_model_config_resolves_selected_model(self):
        config = {
            "model": {"name": "test-model"},
            "models": {"test-model": {"layer_idx": 7, "auc_layer": 5}},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yml"
            config_path.write_text(yaml.safe_dump(config))

            model_id, layer_idx, auc_layer = extract_vectors.load_model_config(
                config_path
            )

        self.assertEqual(model_id, "test-model")
        self.assertEqual(layer_idx, 7)
        self.assertEqual(auc_layer, 5)

    def test_load_model_config_rejects_unknown_model(self):
        config = {
            "model": {"name": "missing-model"},
            "models": {"known-model": {"layer_idx": 7, "auc_layer": 5}},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yml"
            config_path.write_text(yaml.safe_dump(config))

            with self.assertRaisesRegex(ValueError, "not defined"):
                extract_vectors.load_model_config(config_path)

    def test_get_latent_vector_sequential_normalizes_last_token(self):
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = "formatted prompt"
        inputs = type("TokenizedInputs", (dict,), {
            "to": lambda self, device: self,
        })(input_ids=[[1, 2]])
        tokenizer.return_value = inputs

        model = Mock()
        hidden_state = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        model.return_value.hidden_states = [hidden_state]

        vector = extract_vectors.get_latent_vector_sequential(
            "hello", tokenizer, model, torch.device("cpu"), 0
        )

        np.testing.assert_allclose(vector, np.array([0.6, 0.8]))
        tokenizer.apply_chat_template.assert_called_once()
        model.assert_called_once_with(input_ids=[[1, 2]])

    @patch(
        "extract_vectors.Path.resolve",
        return_value=Path("/virtual/project/extract_vectors.py"),
    )
    @patch("extract_vectors.np.savez")
    @patch("extract_vectors.os.makedirs")
    @patch("extract_vectors.get_latent_vector_sequential")
    @patch("extract_vectors.load_prompt_sets")
    @patch("extract_vectors.load_model_and_tokenizer")
    @patch("extract_vectors.load_model_config")
    def test_main_loads_configured_model_and_saves_latent_vectors(
        self,
        load_model_config,
        load_model_and_tokenizer,
        load_prompt_sets,
        get_latent_vector_sequential,
        makedirs,
        savez,
        resolve,
    ):
        load_model_config.return_value = ("test-org/test-model", 7, 5)
        tokenizer, model = Mock(), Mock()
        load_model_and_tokenizer.return_value = (tokenizer, model)
        load_prompt_sets.return_value = (["benign"], ["harmful"])
        get_latent_vector_sequential.side_effect = [
            np.array([0.6, 0.8]),
            np.array([0.8, 0.6]),
        ]

        with patch("builtins.print"):
            result = extract_vectors.main()

        self.assertEqual(result, 0)
        resolve.assert_called_once()
        load_model_config.assert_called_once_with(
            Path("/virtual/project/config/config.yml")
        )
        load_prompt_sets.assert_called_once_with(require_benign=True)
        loaded_model_id, device = load_model_and_tokenizer.call_args.args
        self.assertEqual(loaded_model_id, "test-org/test-model")
        self.assertIsInstance(device, torch.device)
        self.assertEqual(get_latent_vector_sequential.call_count, 2)
        makedirs.assert_called_once_with("./data/test-org", exist_ok=True)
        saved_path = savez.call_args.args[0]
        saved_arrays = savez.call_args.kwargs
        self.assertEqual(
            saved_path, f"./data/{loaded_model_id}_latent_vectors.npz"
        )
        np.testing.assert_array_equal(
            saved_arrays["X_benign"], np.array([[0.6, 0.8]])
        )
        np.testing.assert_array_equal(
            saved_arrays["X_malicious"], np.array([[0.8, 0.6]])
        )
        np.testing.assert_array_equal(
            saved_arrays["prompts_benign"], np.array(["benign"])
        )
        np.testing.assert_array_equal(
            saved_arrays["prompts_malicious"], np.array(["harmful"])
        )


if __name__ == "__main__":
    unittest.main()
