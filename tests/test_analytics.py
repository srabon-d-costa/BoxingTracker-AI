import sys
import unittest
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analytics import (load_analysis, event_timeline,
                       compare_with_ground_truth, normalized_ground_truth)


class DashboardDataTests(unittest.TestCase):
    def test_demo_loads_with_expected_v3_counts(self):
        a = load_analysis(ROOT / 'outputs' / 'demo', duration=13.04)
        self.assertEqual(len(a.events), 30)
        self.assertEqual(a.summary['total'].tolist(), [11, 19])
        self.assertEqual(a.events['confidence'].isna().sum(), 0)

    def test_events_sum_into_timeline(self):
        a = load_analysis(ROOT / 'outputs' / 'demo', duration=13.04)
        timeline = event_timeline(a.events, a.duration)
        self.assertEqual(int(timeline['count'].sum()), 30)
        self.assertEqual(len(timeline), 28)  # 14 seconds x 2 boxers

    def test_duplicate_prediction_counts_as_false_positive(self):
        pred = pd.DataFrame([
            [1.00, 1, 'left'], [1.07, 1, 'left'], [2.00, 2, 'right']
        ], columns=['time_seconds', 'boxer', 'hand'])
        manual = pd.DataFrame([
            [1.02, 1, 'left'], [2.08, 2, 'right'], [3.00, 2, 'left']
        ], columns=['time_seconds', 'boxer', 'hand'])
        r = compare_with_ground_truth(pred, manual, 0.10)
        self.assertEqual((r['tp'], r['fp'], r['fn']), (2, 1, 1))
        self.assertAlmostEqual(r['precision'], 2 / 3)
        self.assertAlmostEqual(r['recall'], 2 / 3)

    def test_different_hands_do_not_match(self):
        pred = pd.DataFrame([[1.00, 1, 'left']], columns=['time_seconds', 'boxer', 'hand'])
        manual = pd.DataFrame([[1.00, 1, 'right']], columns=['time_seconds', 'boxer', 'hand'])
        r = compare_with_ground_truth(pred, manual)
        self.assertEqual((r['tp'], r['fp'], r['fn']), (0, 1, 1))

    def test_empty_and_invalid_labels(self):
        labels = pd.DataFrame(columns=['time_seconds', 'boxer', 'hand'])
        self.assertTrue(normalized_ground_truth(labels).empty)
        invalid = pd.DataFrame([[-1, 4, 'foot']], columns=labels.columns)
        with self.assertRaises(ValueError):
            normalized_ground_truth(invalid)

    def test_only_matching_boxer_matters(self):
        pred = pd.DataFrame([[1.0, 1, 'left']], columns=['time_seconds','boxer','hand'])
        manual = pd.DataFrame([[1.0, 2, 'left']], columns=pred.columns)
        r = compare_with_ground_truth(pred, manual)
        self.assertEqual(r['tp'], 0)


if __name__ == '__main__':
    unittest.main()
