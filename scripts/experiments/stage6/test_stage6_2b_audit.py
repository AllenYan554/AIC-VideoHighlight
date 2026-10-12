"""统计与严格预算的最小独立验收，无真实输入写入。"""
import unittest
import numpy as np
import pandas as pd

from analyze_stage6_2b_native_signals import (
    auc, average_precision, spearman, pooled_auc_kernel, random_interval_mask,
    native_region_mask, measure_mask, edge_diagnostic_flags, exact_budget,
)
from verify_stage6_2b_inputs import project, strict_json


class StatisticsTests(unittest.TestCase):
    def test_auc_ties_and_direction(self):
        # 四个positive-negative pairs: wins=3, tie=1 => 0.875。
        self.assertAlmostEqual(auc([0., 1., 1., 2.], [0, 1, 0, 1]), .875)
        self.assertAlmostEqual(auc([0., -1., -1., -2.], [0, 1, 0, 1]), .125)
        self.assertTrue(np.isnan(auc([1., 1.], [0, 1])))
        self.assertTrue(np.isnan(auc([0., 1.], [1, 1])))

    def test_ap_threshold_ties(self):
        self.assertAlmostEqual(average_precision([2., 1., 1., 0.], [1, 1, 0, 0]), 5 / 6)
        self.assertTrue(np.isnan(average_precision([1., 2.], [0, 0])))

    def test_rank_correlation(self):
        self.assertAlmostEqual(spearman([1, 1, 2, 3], [9, 9, 6, 1]), -1.)
        self.assertTrue(np.isnan(spearman([1, 1, 1], [1, 2, 3])))

    def test_cluster_kernel_matches_expanded_sample(self):
        xs = [np.array([0., .5, 1.]), np.array([.5, .7])]
        ys = [np.array([0, 1, 1]), np.array([0, 1])]
        A, p, n = pooled_auc_kernel(xs, ys)
        for w in [np.array([1, 1]), np.array([3, 1]), np.array([1, 0])]:
            x = np.concatenate([np.tile(v, k) for v, k in zip(xs, w)])
            y = np.concatenate([np.tile(v, k) for v, k in zip(ys, w)])
            self.assertAlmostEqual(float(w @ A @ w / ((w @ p) * (w @ n))), auc(x, y))

    def test_official_like_delta_respects_IoU_mass(self):
        g = pd.DataFrame({"frame": [0, 1, 2], "Y": [1, 1, 0], "Z": [1., .2, 0.]})
        benefit = measure_mask(g, [2], 2, {0}, [])
        harm = measure_mask(g, [0], 2, {0}, [])
        self.assertGreater(benefit["delta_f"], 0)
        self.assertLess(harm["delta_f"], 0)
        self.assertEqual(benefit["z_lost"], 0)


class IdentityBudgetTests(unittest.TestCase):
    def test_projection_exact_half_open(self):
        meta = {"fps": 10., "frame_count": 10, "timestamp_mode": "CFR_FPS"}
        self.assertEqual(project(.11, .3, meta), [2])
        self.assertEqual(project(-.1, 2., meta), list(range(10)))
        self.assertEqual(project(.2, .2, meta), [])

    def test_duplicate_or_nonfinite_json_rejected(self):
        for text in ['{"x":1,"x":2}', '{"x":NaN}']:
            with self.assertRaises(ValueError):
                strict_json(text)

    def test_random_interval_fragmented_exact_unique_budget(self):
        frames = np.array([0, 1, 8, 9, 20, 21])
        rng = np.random.default_rng(13)
        for k in range(7):
            mask, single = random_interval_mask(np.arange(6), frames, k, rng)
            self.assertEqual(len(set(mask)), k)
            self.assertTrue(set(mask) <= set(range(6)))
            if k > 2:
                self.assertFalse(single)

    def test_overlapping_parents_missing_scores_budget(self):
        frames = np.arange(8)
        eligible = np.array([0, 1, 3, 4, 6, 7])
        scores = np.array([.2, .2, np.nan, .1, .1, np.nan, .8, .8])
        regions = [np.arange(5), np.arange(3, 8)]
        hashes = np.arange(8)[::-1].astype(np.uint64)
        for k in range(7):
            mask = native_region_mask(eligible, frames, scores, hashes, regions, k)
            self.assertEqual(len(set(mask)), k)
            self.assertTrue(set(mask) <= set(eligible))
        self.assertEqual(exact_budget(31, .1), 3)

    def test_multi_parent_keep_union_and_no_core(self):
        E = np.array([2., 2., 2.])
        S = np.array([1., 2., 1.])
        self.assertEqual(edge_diagnostic_flags(E, S, [{0, 1, 2}], 1).tolist(), [1., 0., 1.])
        # 第二parent无core，KEEP覆盖2，union必须保守保留2。
        self.assertEqual(edge_diagnostic_flags(E, S, [{0, 1, 2}, {2}], 1).tolist(), [1., 0., 0.])


if __name__ == "__main__":
    unittest.main()
