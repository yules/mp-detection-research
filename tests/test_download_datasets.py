import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import download_datasets


class DownloadDatasetsTests(unittest.TestCase):
    @patch("download_datasets.os.makedirs")
    @patch("download_datasets.load_dataset")
    def test_download_and_save_datasets_saves_harmbench(
        self, load_dataset, makedirs
    ):
        dataset = Mock()
        load_dataset.return_value = dataset
        base_dir = "./test-data"

        download_datasets.download_and_save_datasets(base_dir)

        load_dataset.assert_called_once_with(
            "walledai/HarmBench", "standard", trust_remote_code=True
        )
        dataset.save_to_disk.assert_called_once_with(
            str(Path(base_dir) / "harmbench")
        )
        makedirs.assert_called_once_with(base_dir, exist_ok=True)

    @patch("download_datasets.os.makedirs")
    @patch("download_datasets.load_dataset", side_effect=RuntimeError("network error"))
    def test_download_and_save_datasets_handles_download_failure(
        self, load_dataset, makedirs
    ):
        download_datasets.download_and_save_datasets("./test-data")

        load_dataset.assert_called_once()
        makedirs.assert_called_once_with("./test-data", exist_ok=True)


if __name__ == "__main__":
    unittest.main()
