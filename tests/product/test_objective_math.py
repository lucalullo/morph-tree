"""Independent zero-fit derivatives and metric references from retained math tests.

These checks qualify objective arithmetic, not predictive superiority or training.
Softmax uses a diagonal majorizer, not the exact full Hessian.
"""
import math
import random
import unittest

from morphtree.backend import PythonBackend
from morphtree.metrics import softmax, multiclass_metrics, rmse, r2_score


class ObjectiveMathContracts(unittest.TestCase):
    def test_softmax_shift_extremes_and_simplex(self):
        for row in ([0., 1., -2.], [-10000., 10000., 0.], [10000.] * 100):
            p = softmax(row)
            self.assertAlmostEqual(math.fsum(p), 1.)
            self.assertTrue(all(math.isfinite(v) and 0 <= v <= 1 for v in p))
            for a, b in zip(p, softmax([v + 123. for v in row])):
                self.assertAlmostEqual(a, b, places=13)

    def test_softmax_gradient_full_hessian_and_diagonal_majorizer(self):
        rng, backend, eps = random.Random(101), PythonBackend(), 1e-5
        for k in (2, 3, 7):
            z = [rng.uniform(-2, 2) for _ in range(k)]
            p = softmax(z)
            def loss(v):
                maximum = max(v)
                return maximum - v[1] + math.log(math.fsum(math.exp(x - maximum) for x in v))
            for j in range(k):
                plus, minus = list(z), list(z)
                plus[j] += eps
                minus[j] -= eps
                g, h = backend.multiclass_gradient_column([p], [1], j, 'softmax')
                self.assertAlmostEqual(g[0], (loss(plus) - loss(minus)) / (2 * eps), places=8)
                self.assertAlmostEqual(h[0], 2 * p[j] * (1 - p[j]), places=14)
                for i in range(k):
                    derivative = (softmax(plus)[i] - softmax(minus)[i]) / (2 * eps)
                    self.assertAlmostEqual(derivative, p[i] * ((i == j) - p[j]), places=8)
            self.assertAlmostEqual(math.fsum(p[i] - (i == 1) for i in range(k)), 0.)
            for _ in range(30):
                v = [rng.gauss(0, 1) for _ in range(k)]
                exact = math.fsum(p[i] * v[i] ** 2 for i in range(k)) - math.fsum(a*b for a,b in zip(p,v)) ** 2
                bound = math.fsum(2*p[i]*(1-p[i])*v[i]**2 for i in range(k))
                self.assertGreaterEqual(bound + 1e-13, exact)

    def test_multiclass_metrics_known_confusion_and_missing_class(self):
        got = multiclass_metrics([0, 1, 2, 2], [[.8,.1,.1],[.1,.8,.1],[.1,.8,.1],[.1,.1,.8]])
        self.assertEqual(got['accuracy'], .75)
        self.assertAlmostEqual(got['macro_f1'], 7/9)
        self.assertAlmostEqual(got['logloss'], -(3*math.log(.8)+math.log(.1))/4)
        self.assertEqual(multiclass_metrics([0], [[1.,0.,0.]])['macro_f1'], 1/3)

    def test_logistic_derivatives_against_finite_differences(self):
        backend, eps = PythonBackend(), 1e-4
        for x in (-2., -.2, 1.7):
            for y in (0., 1.):
                loss = lambda z: math.log1p(math.exp(z))-y*z
                _, g, h = backend.gradients([x], [y])
                self.assertAlmostEqual(g[0], (loss(x+eps)-loss(x-eps))/(2*eps), places=8)
                self.assertAlmostEqual(h[0], (loss(x+eps)+loss(x-eps)-2*loss(x))/(eps*eps), places=7)

    def test_regression_derivatives_and_metrics_across_scales(self):
        backend = PythonBackend()
        for scale in (1e-6, 1., 1e6):
            prediction, target = [-.4*scale, 2.*scale], [-2.*scale, -.5*scale]
            _, g, h = backend.regression_gradients(prediction, target)
            eps = scale * 1e-3
            for i, (x, y) in enumerate(zip(prediction, target)):
                loss = lambda z: .5*(z-y)**2
                self.assertAlmostEqual(g[i]/scale, (loss(x+eps)-loss(x-eps))/(2*eps*scale), places=8)
                self.assertAlmostEqual(h[i], (loss(x+eps)+loss(x-eps)-2*loss(x))/(eps*eps), places=6)
            self.assertAlmostEqual(rmse(target, prediction)/scale, math.sqrt(8.81/2), places=12)
            self.assertAlmostEqual(r2_score(target, prediction), 1-8.81/1.125, places=12)
        self.assertEqual(r2_score([2., 2.], [2., 2.]), 1.)
        self.assertEqual(r2_score([2., 2.], [1., 1.]), 0.)
