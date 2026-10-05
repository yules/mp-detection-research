import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import multi_vector_repe
from prev_detection_scripts import detect_anomalies, repe_detect


class DetectionScriptsTests(unittest.TestCase):
    def test_repe_status_labels_cover_prediction_outcomes(self):
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
            with self.subTest(row=row):
                self.assertEqual(repe_detect.evaluate_status(row), expected)

    @patch("multi_vector_repe.Path.resolve", return_value=Path("/virtual/project/multi_vector_repe.py"))
    @patch("multi_vector_repe.pd.DataFrame.to_csv")
    @patch("multi_vector_repe.get_latent_vector_sequential")
    @patch("multi_vector_repe.load_model_and_tokenizer")
    @patch("multi_vector_repe.load_stress_test_prompts")
    @patch("multi_vector_repe.load_latent_vectors")
    @patch("multi_vector_repe.load_model_config")
    def test_multi_vector_repe_reads_configured_model_vectors(
        self,
        load_model_config,
        load_latent_vectors,
        load_stress_test_prompts,
        load_model_and_tokenizer,
        get_latent_vector_sequential,
        to_csv,
        resolve,
    ):
        load_model_config.return_value = ("test-model-repe", 12, 11)
        rng = np.random.RandomState(42)
        n_benign, n_malicious, dim = 60, 60, 16
        fake_data = {
            "X_benign": rng.randn(n_benign, dim),
            "X_malicious": rng.randn(n_malicious, dim),
            "prompts_benign": np.array([f"b{i}" for i in range(n_benign)]),
            "prompts_malicious": np.array([f"m{i}" for i in range(n_malicious)]),
        }
        load_latent_vectors.return_value = tuple(fake_data.values())
        load_stress_test_prompts.return_value = (["gcg-1"], ["roleplay-1"])
        load_model_and_tokenizer.return_value = (Mock(), Mock())
        get_latent_vector_sequential.return_value = rng.randn(dim)

        with patch("builtins.print"):
            multi_vector_repe.main()

        load_model_config.assert_called_once_with(
            Path("/virtual/project/config/config.yml")
        )
        load_latent_vectors.assert_called_once_with("test-model-repe")
        to_csv.assert_called_once()

    @patch("prev_detection_scripts.repe_detect.pd.DataFrame.to_csv")
    @patch("prev_detection_scripts.repe_detect.load_latent_vectors")
    @patch("prev_detection_scripts.repe_detect.load_model_config")
    def test_repe_detect_reads_configured_model_vectors(
        self,
        load_model_config,
        load_latent_vectors,
        to_csv,
    ):
        load_model_config.return_value = ("test-model-single", 8, 7)
        rng = np.random.RandomState(42)
        n_benign, n_malicious, dim = 60, 60, 16
        fake_data = {
            "X_benign": rng.randn(n_benign, dim),
            "X_malicious": rng.randn(n_malicious, dim),
            "prompts_benign": np.array([f"b{i}" for i in range(n_benign)]),
            "prompts_malicious": np.array([f"m{i}" for i in range(n_malicious)]),
        }
        load_latent_vectors.return_value = tuple(fake_data.values())

        with patch("builtins.print"):
            repe_detect.main()

        load_latent_vectors.assert_called_once_with("test-model-single")
        to_csv.assert_called_once()

    @patch("prev_detection_scripts.detect_anomalies.pd.DataFrame.to_csv")
    @patch("prev_detection_scripts.detect_anomalies.load_latent_vectors")
    @patch("prev_detection_scripts.detect_anomalies.load_model_config")
    def test_detect_anomalies_reads_configured_model_vectors(
        self,
        load_model_config,
        load_latent_vectors,
        to_csv,
    ):
        load_model_config.return_value = ("test-model-mahalanobis", 4, 3)
        rng = np.random.RandomState(42)
        n_benign, n_malicious, dim = 200, 50, 160
        fake_data = {
            "X_benign": rng.randn(n_benign, dim),
            "X_malicious": rng.randn(n_malicious, dim),
            "prompts_benign": np.array([f"b{i}" for i in range(n_benign)]),
            "prompts_malicious": np.array([f"m{i}" for i in range(n_malicious)]),
        }
        load_latent_vectors.return_value = tuple(fake_data.values())

        with patch("builtins.print"):
            detect_anomalies.main()

        load_latent_vectors.assert_called_once_with("test-model-mahalanobis")
        to_csv.assert_called_once()


if __name__ == "__main__":
    unittest.main()
