import os
import unittest
from unittest.mock import Mock, call, patch

import download_datasets


class DownloadDatasetsTests(unittest.TestCase):
    @patch("download_datasets.os.makedirs")
    @patch("download_datasets.pd.read_csv")
    @patch("download_datasets.load_dataset")
    def test_download_and_save_datasets_saves_all_datasets(
        self, load_dataset, read_csv, makedirs
    ):
        datasets = [Mock(), Mock(), Mock()]
        load_dataset.side_effect = datasets
        advbench = Mock()
        read_csv.return_value = advbench
        base_dir = "./test-data"

        with patch("builtins.print"):
            download_datasets.download_and_save_datasets(base_dir)

        self.assertEqual(
            load_dataset.call_args_list,
            [
                call("walledai/HarmBench", "standard", trust_remote_code=True),
                call("FloofCat/AdvSuffixes"),
                call("rubend18/ChatGPT-Jailbreak-Prompts"),
            ],
        )
        read_csv.assert_called_once_with(download_datasets.ADVBENCH_URL)
        advbench.to_csv.assert_called_once_with(
            os.path.join(base_dir, "advbench.csv"), index=False
        )
        for dataset, local_name in zip(
            datasets, ["harmbench", "adv_suffixes", "jailbreak_prompts"]
        ):
            dataset.save_to_disk.assert_called_once_with(
                os.path.join(base_dir, local_name)
            )
        makedirs.assert_called_once_with(base_dir, exist_ok=True)

    @patch("download_datasets.os.makedirs")
    @patch("download_datasets.pd.read_csv", side_effect=RuntimeError("network error"))
    @patch("download_datasets.load_dataset")
    def test_download_and_save_datasets_continues_after_failure(
        self, load_dataset, read_csv, makedirs
    ):
        saved_datasets = [Mock(), Mock()]
        load_dataset.side_effect = [RuntimeError("network error"), *saved_datasets]

        with patch("builtins.print"):
            download_datasets.download_and_save_datasets("./test-data")

        self.assertEqual(load_dataset.call_count, 3)
        read_csv.assert_called_once_with(download_datasets.ADVBENCH_URL)
        for dataset in saved_datasets:
            dataset.save_to_disk.assert_called_once()
        makedirs.assert_called_once_with("./test-data", exist_ok=True)


if __name__ == "__main__":
    unittest.main()
