import unittest

import pandas as pd

from logcluster import discover_patterns, extract_log_text, normalize_line, read_dataset


class NormalizeLineTests(unittest.TestCase):
    def test_normalizes_timestamps_ips_and_numbers(self):
        first = normalize_line("2025-03-08 10:42:11 ERROR host=10.0.0.1 retry=4")
        second = normalize_line("2025-04-09 11:53:22 ERROR host=10.0.0.2 retry=9")
        self.assertEqual(first, second)
        self.assertIn("<TIMESTAMP>", first)
        self.assertIn("<IP>", first)
        self.assertIn("<NUM>", first)


class DiscoverPatternsTests(unittest.TestCase):
    def test_repeated_normalized_messages_form_one_pattern(self):
        text = "\n".join([
            "ERROR Timeout connecting to 10.0.0.1",
            "ERROR Timeout connecting to 10.0.0.2",
        ])
        line_count, patterns = discover_patterns(text)
        self.assertEqual(line_count, 2)
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0].count, 2)
        self.assertIn("<IP>", patterns[0].template)

    def test_singleton_messages_remain_separate_rare_patterns(self):
        line_count, patterns = discover_patterns(
            "CRITICAL kernel memory failure\n"
            "WARN certificate expires tomorrow"
        )
        self.assertEqual(line_count, 2)
        self.assertEqual(len(patterns), 2)
        self.assertEqual([pattern.count for pattern in patterns], [1, 1])

    def test_empty_input_returns_no_patterns(self):
        self.assertEqual(discover_patterns("\n  \n"), (0, []))

    def test_rejects_invalid_similarity(self):
        with self.assertRaises(ValueError):
            discover_patterns("INFO ready", similarity=1.0)


class DatasetTests(unittest.TestCase):
    def test_reads_csv_and_extracts_nonempty_message_values(self):
        dataset = read_dataset(
            "events.csv",
            b"timestamp,message\n"
            b"2025-01-01,ERROR timeout to 10.0.0.1\n"
            b"2025-01-02,ERROR timeout to 10.0.0.2\n"
            b"2025-01-03,\n",
        )
        log_text = extract_log_text(dataset, "message")
        self.assertEqual(log_text, "ERROR timeout to 10.0.0.1\nERROR timeout to 10.0.0.2")
        line_count, patterns = discover_patterns(log_text)
        self.assertEqual(line_count, 2)
        self.assertEqual(patterns[0].count, 2)

    def test_reads_jsonl_records(self):
        dataset = read_dataset(
            "events.jsonl",
            b'{"message":"INFO ready"}\n{"message":"WARN retry"}\n',
        )
        self.assertEqual(extract_log_text(dataset, "message"), "INFO ready\nWARN retry")

    def test_reads_json_records(self):
        dataset = read_dataset(
            "events.json",
            b'[{"message":"INFO ready"},{"message":"WARN retry"}]',
        )
        self.assertEqual(extract_log_text(dataset, "message"), "INFO ready\nWARN retry")

    def test_rejects_unsupported_dataset_type_and_column(self):
        with self.assertRaises(ValueError):
            read_dataset("events.xlsx", b"")
        with self.assertRaises(ValueError):
            extract_log_text(pd.DataFrame({"message": ["INFO ready"]}), "missing")


if __name__ == "__main__":
    unittest.main()
