#!/usr/bin/env python3

import base64
import json
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from prepare import (  # noqa: E402
    TOKEN_IDS_ENCODING,
    TOKEN_SCHEMA_VERSION,
    _prepare_record,
    main as prepare_main,
    save_shard,
)


def encoded(ids):
    raw = np.asarray(ids, dtype="<u4").tobytes()
    return base64.b64encode(zlib.compress(raw)).decode("ascii")


def record(prompt_ids, response_ids):
    return {
        "group_id": 17,
        "sample_id": 3,
        "prompt": "This text is deliberately unrelated to the stored IDs.",
        "response": "It must never be tokenized by prepare.py.",
        "token_schema_version": TOKEN_SCHEMA_VERSION,
        "token_ids_encoding": TOKEN_IDS_ENCODING,
        "prompt_token_ids_b64": encoded(prompt_ids),
        "response_token_ids_b64": encoded(response_ids),
        "prompt_token_count": len(prompt_ids),
        "response_token_count": len(response_ids),
        "requested_max_tokens": len(response_ids),
        "generation_hit_max_tokens": True,
        "generation_finish_reason": None,
        "generation_stop_reason": None,
        "reward_score": 1.0,
    }


class TokenPipelineTest(unittest.TestCase):
    def test_prepare_uses_stored_ids_and_builds_shifted_response_mask(self):
        entry = _prepare_record((9, record([10, 11], [20, 21, 22, 23])), 5)
        np.testing.assert_array_equal(entry["input_ids"], [10, 11, 20, 21, 22])
        np.testing.assert_array_equal(entry["attention_mask"], [1, 1, 1, 1, 1])
        np.testing.assert_array_equal(entry["response_mask"], [0, 1, 1, 1])
        self.assertEqual(entry["prompt_length"], 2)
        self.assertEqual(entry["generated_response_length"], 4)
        self.assertEqual(entry["original_input_length"], 6)
        self.assertTrue(entry["was_truncated"])
        self.assertTrue(entry["generation_hit_max_tokens"])

    def test_text_only_rollout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be regenerated"):
            _prepare_record(
                (0, {"group_id": 1, "sample_id": 0, "reward_score": 0.0}),
                32,
            )

    def test_declared_token_count_must_match_payload(self):
        bad_record = record([1, 2], [3])
        bad_record["response_token_count"] = 2
        with self.assertRaisesRegex(ValueError, "inconsistent response token count"):
            _prepare_record((0, bad_record), 32)

    def test_shard_preserves_generation_metadata(self):
        entry = _prepare_record((0, record([1, 2], [3, 4])), 32)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "data_rank_0.npz")
            save_shard(path, [entry])
            with np.load(path, allow_pickle=True) as archive:
                self.assertEqual(int(archive["prompt_length"][0]), 2)
                self.assertEqual(int(archive["generated_response_length"][0]), 2)
                self.assertTrue(bool(archive["generation_hit_max_tokens"][0]))

    def test_prepare_main_writes_schema_v3_group_shards(self):
        rows = []
        for group_id in (10, 20):
            for sample_id in (0, 1):
                row = record([1, 2], [3, 4])
                row["group_id"] = group_id
                row["sample_id"] = sample_id
                rows.append(row)
        with tempfile.TemporaryDirectory() as directory:
            responses_path = Path(directory) / "responses_sorted.json"
            responses_path.write_text(json.dumps(rows), encoding="utf-8")
            argv = [
                "prepare.py",
                "--responses_dir",
                directory,
                "--model_path",
                str(Path(directory) / "model"),
                "--world_size",
                "2",
                "--rollout_n",
                "2",
                "--max_length",
                "32",
                "--num_workers",
                "1",
            ]
            original_argv = sys.argv
            try:
                sys.argv = argv
                prepare_main()
            finally:
                sys.argv = original_argv
            manifest_path = (
                Path(directory) / "data_independent" / "partition_manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["partition_schema_version"], 3)
            self.assertEqual(manifest["token_input_source"], "inference_token_ids")
            self.assertEqual(manifest["expected_groups"], 2)
            self.assertEqual(manifest["expected_responses"], 4)
            self.assertTrue(
                (Path(directory) / "data_independent" / "data_rank_0.npz").is_file()
            )
            self.assertTrue(
                (Path(directory) / "data_independent" / "data_rank_1.npz").is_file()
            )


if __name__ == "__main__":
    unittest.main()
