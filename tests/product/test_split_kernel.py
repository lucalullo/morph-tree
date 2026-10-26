"""Canonical specialization checked against the retained declarative interpreter."""
from dataclasses import replace
import importlib.util
import math
import random
import unittest
from morphtree import AlgorithmGenome, HyperParams
from morphtree.backend import PythonBackend
from morphtree.tree import HistogramTree


class SplitKernelTests(unittest.TestCase):
    def test_random_histograms_match_generic_interpreter_exactly(self):
        rng = random.Random(6901)
        for bins in (2, 8, 32, 257):
            for missing in (False, True):
                for policy in ('learned', 'left', 'right'):
                    for regularization in ('global', 'depth_scaled'):
                        hist = [[float(rng.randrange(2, 12)) for _ in range(bins)],
                                [rng.uniform(-3, 3) for _ in range(bins)],
                                [rng.uniform(.001, 3) for _ in range(bins)],
                                [rng.random() for _ in range(bins)],
                                [rng.random() for _ in range(bins)]]
                        if not missing:
                            for row in hist: row[0] = 0.
                        p = HyperParams(min_samples_leaf=2, min_gain=0., l1=.25, l2=.5, depth_reg_strength=.7)
                        g = AlgorithmGenome(missing_policy=policy, regularization_policy=regularization)
                        generic = replace(g, split_program=('OBJECTIVE_GAIN', 'OBJECTIVE_GAIN', 'MIN'))
                        expected = HistogramTree(generic, p)._best_split({0: hist}, 2, [0])
                        actual = HistogramTree(g, p)._best_split({0: hist}, 2, [0])
                        self.assertEqual(actual, expected)

    def test_admission_ties_and_rounding_use_scalar_reference(self):
        hist = [[0., 10., 10., 10.], [0., -1., 0., 1.], [0., 2., 2., 2.], [0.]*4, [0.]*4]
        p = HyperParams(min_samples_leaf=1, l1=0., min_gain=0.)
        g = AlgorithmGenome()
        total = [math.fsum(r) for r in hist]
        gain = PythonBackend().fast_split(hist, total, 0, g, p)[0]
        for threshold in (0., gain, math.nextafter(gain, 0.), math.nextafter(gain, math.inf)):
            pp = replace(p, min_gain=threshold)
            reference = replace(g, split_program=('OBJECTIVE_GAIN','OBJECTIVE_GAIN','MIN'))
            self.assertEqual(HistogramTree(g, pp)._best_split({0:hist,1:hist},0,[0,1]),
                             HistogramTree(reference, pp)._best_split({0:hist,1:hist},0,[0,1]))

    @unittest.skipUnless(importlib.util.find_spec('numpy'), 'NumPy optional')
    def test_forced_numpy_fallback_is_exact(self):
        import numpy as np
        from unittest.mock import patch
        from morphtree.numpy_backend import NumpyBackend
        hist = np.array([[3.,9.,11.,7.],[-1.,-2.,.5,2.],[.6,2.,3.,1.],[0.]*4,[0.]*4])
        p = HyperParams(min_samples_leaf=2, min_gain=0.)
        g = AlgorithmGenome()
        expected = HistogramTree(replace(g,split_program=('OBJECTIVE_GAIN','OBJECTIVE_GAIN','MIN')),p)._best_split({0:hist},1,[0])
        with patch.object(NumpyBackend,'fast_split',return_value=NotImplemented):
            self.assertEqual(HistogramTree(g,p,NumpyBackend())._best_split({0:hist},1,[0]),expected)


if __name__ == '__main__': unittest.main()
