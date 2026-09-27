import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

import extract_all_layers


class HarmBenchTrain(dict):
    column_names = ["Behavior"]


class ExtractAllLayersTests(unittest.TestCase):
    @patch("extract_all_layers.tokenize_prompt")
    def test_get_all_layer_vectors_normalizes_final_token_of_each_layer(
        self, tokenize_prompt
    ):
        inputs = {"input_ids": [[1, 2]]}
        tokenize_prompt.return_value = inputs
        model = Mock()
        model.return_value.hidden_states = [
            torch.tensor([[[9.0, 9.0], [3.0, 4.0]]]),
            torch.tensor([[[8.0, 8.0], [5.0, 12.0]]]),
        ]
        tokenizer = Mock()
        device = torch.device("cpu")

        vectors = extract_all_layers.get_all_layer_vectors(
            "hello", tokenizer, model, device
        )

        np.testing.assert_allclose(
            vectors,
            np.array([[0.6, 0.8], [5.0 / 13.0, 12.0 / 13.0]]),
        )
        self.assertEqual(vectors.shape, (2, 2))
        tokenize_prompt.assert_called_once_with("hello", tokenizer, device)
        model.assert_called_once_with(**inputs)

    @patch("extract_all_layers.np.savez")
    @patch("extract_all_layers.os.makedirs")
    @patch("extract_all_layers.get_all_layer_vectors")
    @patch("extract_all_layers.load_from_disk")
    @patch("extract_all_layers.load_model_and_tokenizer")
    @patch("extract_all_layers.load_model_config")
    def test_main_loads_configured_model_and_saves_layer_vectors(
        self,
        load_model_config,
        load_model_and_tokenizer,
        load_from_disk,
        get_all_layer_vectors,
        makedirs,
        savez,
    ):
        load_model_config.return_value = ("configured-model", 5)
        tokenizer, model = Mock(), Mock()
        load_model_and_tokenizer.return_value = (tokenizer, model)
        load_from_disk.side_effect = [
            {"train": HarmBenchTrain(Behavior=["harmful"])},
            {"messages": [[{"content": "benign"}]]},
        ]
        get_all_layer_vectors.side_effect = [
            np.array([[1.0, 0.0]]),
            np.array([[0.0, 1.0]]),
        ]

        with patch("builtins.print"):
            result = extract_all_layers.main()

        self.assertEqual(result, 0)
        load_model_config.assert_called_once()
        loaded_model_id, device = load_model_and_tokenizer.call_args.args
        self.assertEqual(loaded_model_id, "configured-model")
        self.assertIsInstance(device, torch.device)
        self.assertEqual(get_all_layer_vectors.call_count, 2)
        makedirs.assert_called_once_with("./data", exist_ok=True)
        saved_path = savez.call_args.args[0]
        saved_arrays = savez.call_args.kwargs
        self.assertEqual(saved_path, "./data/llama_all_layers.npz")
        np.testing.assert_array_equal(
            saved_arrays["benign_states"], np.array([[[1.0, 0.0]]])
        )
        np.testing.assert_array_equal(
            saved_arrays["malicious_states"], np.array([[[0.0, 1.0]]])
        )


if __name__ == "__main__":
    unittest.main()