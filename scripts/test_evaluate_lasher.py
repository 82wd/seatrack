"""Synthetic protocol checks; these do not substitute for MATLAB parity testing."""
import unittest
import numpy as np
from evaluate_lasher import sequence_curves, summarize


class ProtocolTests(unittest.TestCase):
    def test_perfect_and_strict_iou_boundary(self):
        gt = np.array([[10., 10., 40., 40.]] * 3)
        curves, _ = sequence_curves(gt, gt)
        scores, _, _ = summarize([curves])
        self.assertEqual(scores['PR'], 1)
        self.assertEqual(scores['NPR'], 1)
        self.assertAlmostEqual(scores['SR'], 20 / 21)  # IoU > 1 is false

    def test_threshold_equality_and_first_frame(self):
        # Centers / width are exactly 0.5 and 0.75; avoid decimal-boundary
        # rounding when checking the <= predicate itself.
        gt = np.array([[.5, 10., 80., 80.]] * 2)
        pred = gt.copy()
        pred[:, 0] += 20
        original = pred.copy()
        curves, _ = sequence_curves(pred, gt)
        self.assertEqual(curves['precision'][19], .5)
        self.assertEqual(curves['precision'][20], 1)
        self.assertEqual(curves['normalized_precision'][24], .5)
        self.assertEqual(curves['normalized_precision'][25], 1)
        np.testing.assert_array_equal(pred, original)

    def test_invalid_gt_matches_reference_minus_one(self):
        gt = np.array([[10., 10., 40., 40.], [0., 10., 40., 40.]])
        curves, counts = sequence_curves(gt, gt)
        self.assertEqual(counts['invalid_gt_frames'], 1)
        self.assertEqual(curves['precision'][0], 1)
        self.assertEqual(curves['success'][0], .5)

    def test_carry_forward(self):
        gt = np.array([[10., 10., 40., 40.]] * 3)
        pred = gt.copy()
        pred[1:, 2] = 0
        curves, counts = sequence_curves(pred, gt)
        self.assertEqual(counts['carried_forward_predictions'], 2)
        self.assertEqual(curves['precision'][0], 1)

    def test_equal_sequence_weight(self):
        short_gt = np.array([[10., 10., 40., 40.]] * 2)
        long_gt = np.array([[10., 10., 40., 40.]] * 10)
        long_pred = long_gt.copy()
        long_pred[1:, 0] += 1000
        first, _ = sequence_curves(short_gt, short_gt)
        second, _ = sequence_curves(long_pred, long_gt)
        scores, _, _ = summarize([first, second])
        self.assertAlmostEqual(scores['PR'], .55)

    def test_zero_curve_exclusion(self):
        curves = {k: np.array([1.] * n) for k, n in
                  [('precision', 51), ('normalized_precision', 51), ('success', 21)]}
        zeros = {k: np.zeros_like(v) for k, v in curves.items()}
        scores, _, excluded = summarize([curves, zeros])
        self.assertEqual(scores['SR'], 1)
        self.assertEqual(excluded['success'], [1])

    def test_rejects_length_mismatch(self):
        with self.assertRaises(ValueError):
            sequence_curves(np.ones((2, 4)), np.ones((3, 4)))


if __name__ == '__main__':
    unittest.main()
