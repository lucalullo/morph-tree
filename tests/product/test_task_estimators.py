"""Task facade contracts only; synthetic delegates never count as LIVE fits."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import morphtree
from morphtree import MorphTree, MorphTreeClassifier, MorphTreeRegressor, model
from morphtree import api
from morphtree._experimental.admission import AdmissionError
from morphtree._experimental.target import detect_target


class TaskEstimatorContracts(unittest.TestCase):
    def test_public_exports_and_native_dispatch_are_separate(self):
        from morphtree._experimental.training import NATIVE
        self.assertIs(NATIVE['binary'], model.MorphTreeClassifier)
        self.assertIs(NATIVE['regression'], model.MorphTreeRegressor)
        self.assertIs(morphtree.NativeMorphTreeClassifier, model.MorphTreeClassifier)
        self.assertIs(morphtree.NativeMorphTreeRegressor, model.MorphTreeRegressor)
        self.assertIsNot(MorphTreeClassifier, model.MorphTreeClassifier)
        self.assertIn('MorphTreeClassifier', morphtree.__all__)
        self.assertIn('MorphTreeRegressor', morphtree.__all__)

    def test_both_task_facades_share_the_evolving_engine(self):
        for cls, task in ((MorphTreeClassifier, 'classification'), (MorphTreeRegressor, 'regression')):
            value = cls(random_state=42)
            self.assertIs(type(value._engine), MorphTree)
            self.assertEqual(value._engine.task, task)
            self.assertTrue(value._engine._evolving_program)
            self.assertEqual(value._engine.tuning, 'optuna')
            self.assertEqual(value.max_mutations, 2)
            with self.assertRaises(TypeError):
                cls(task=task)

    def test_classifier_resolves_only_binary_or_multiclass(self):
        intent = MorphTreeClassifier()._engine.task
        for y, expected in ((['no','yes']*2, 'binary'), ([.1,.2]*2, 'binary'),
                            ([.1,.2,.3,.4], 'multiclass'), ([1,2,3,4], 'multiclass')):
            with self.subTest(y=y):
                task, labels, classes = detect_target(y, intent)
                self.assertEqual(task, expected)
                self.assertEqual([classes[i] for i in labels], y)

    def test_regression_does_not_infer_classification(self):
        for y in ([0,1,0,1], [1,2,3,4], [.1,.2,.3,.4]):
            task, labels, classes = detect_target(y, MorphTreeRegressor()._engine.task)
            self.assertEqual((task, labels, classes), ('regression', y, []))
        with self.assertRaisesRegex(ValueError, 'numeric'):
            detect_target(['a','b']*2, 'regression')

    def test_bad_classification_targets_are_rejected(self):
        for y in ([1]*4, [1,2,None,1], [1,2,float('nan'),1], [1,'b',1,'b'], [[1]]*4):
            with self.subTest(y=y), self.assertRaises(ValueError):
                detect_target(y, 'classification')

    def test_controls_roundtrip_without_hidden_task_parameter(self):
        for cls in (MorphTreeClassifier, MorphTreeRegressor):
            value = cls(random_state=42, verbosity=1, use_optuna=False,
                max_mutations=3, time_budget=40, n_trials=1, capacity={'max_depth':3})
            params = value.get_params()
            self.assertNotIn('task', params)
            self.assertEqual(cls(**params).get_params(), params)
            params['capacity']['max_depth'] = 8
            self.assertEqual(value.capacity, {'max_depth':3})
            self.assertTrue(value._engine._evolving_program)

    def test_invalid_set_params_preserves_previous_engine_and_weights(self):
        value = MorphTreeClassifier(max_mutations=3)
        engine = value._engine
        engine.model_ = object()  # State sentinel only, not trained weights.
        for options in ({'task':'regression'}, {'use_optuna':'yes'}, {'max_mutations':9}, {'n_jobs':2}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                value.set_params(**options)
            self.assertIs(value._engine, engine)
            self.assertTrue(hasattr(engine, 'model_'))

    def test_valid_set_params_resets_state_and_preserves_other_controls(self):
        value = MorphTreeClassifier(random_state=42, max_mutations=3)
        value._engine.model_ = object()
        self.assertIs(value.set_params(use_optuna=False), value)
        self.assertEqual(value.max_mutations, 3)
        self.assertEqual(value.random_state, 42)
        self.assertEqual(value._engine.tuning, 'none')
        self.assertFalse(hasattr(value._engine, 'model_'))

    def test_fit_delegates_once_and_returns_facade_with_original_objects(self):
        X, y = object(), object()
        for cls in (MorphTreeClassifier, MorphTreeRegressor):
            value = cls()
            with patch.object(MorphTree, 'fit', autospec=True, return_value=value._engine) as fake:
                self.assertIs(value.fit(X, y), value)
                fake.assert_called_once_with(value._engine, X, y)

    def test_input_guards_precede_target_materialization(self):
        class Unreadable:
            def tolist(self):
                raise AssertionError('Target read before context guard')
        value = MorphTreeClassifier(use_optuna=False)
        with patch.object(api, 'PUBLIC_SCOPE') as scope, patch.object(api, 'ACTIVE') as active:
            scope.get.return_value = None
            active.get.return_value = object()
            with self.assertRaises(AdmissionError):
                api._new_user_session(value._engine, Unreadable(), Unreadable())
        class Sealed(Unreadable):
            _morphtree_sealed = True
        with self.assertRaisesRegex(ValueError, 'Controlled-source'):
            api._new_user_session(value._engine, Sealed(), Sealed())

    def test_loaded_model_task_checked_and_classifier_refit_intent_preserved(self):
        for task in ('binary','multiclass','regression'):
            engine = MorphTree(task=task, use_optuna=False)
            engine.task_ = task
            wanted = MorphTreeRegressor if task == 'regression' else MorphTreeClassifier
            wrong = MorphTreeClassifier if task == 'regression' else MorphTreeRegressor
            with patch.object(MorphTree, 'load', return_value=engine) as fake:
                loaded = wanted.load('synthetic-loader-fixture.json', sha256='digest', workdir='work')
                fake.assert_called_once_with('synthetic-loader-fixture.json', sha256='digest', workdir='work')
                self.assertIs(loaded._engine, engine)
                self.assertEqual(engine.task, wanted._task_kind)
                with self.assertRaisesRegex(ValueError, 'incompatible'):
                    wrong.load('synthetic-loader-fixture.json')

    def test_fitted_attributes_are_defensive_and_regressor_has_no_probabilities(self):
        value = MorphTreeClassifier()
        value._engine.classes_ = ['a','b']
        value._engine.n_features_in_ = 3
        classes = value.classes_; classes.append('c')
        self.assertEqual(value.classes_, ['a','b'])
        self.assertEqual(value.n_features_in_, 3)
        self.assertFalse(hasattr(MorphTreeRegressor(), 'predict_proba'))
        with self.assertRaises(AttributeError):
            MorphTreeRegressor().classes_

    def test_task_specific_scores_and_invalid_score_targets(self):
        classifier, regressor = MorphTreeClassifier(), MorphTreeRegressor()
        with patch.object(classifier, 'predict', return_value=['a','b','b']):
            self.assertEqual(classifier.score(None, ['a','a','b']), 2/3)
            for y in ([], ['a'], ['a',None,'b']):
                with self.assertRaises(ValueError):
                    classifier.score(None, y)
        with patch.object(regressor, 'predict', return_value=[1.,2.,3.]):
            self.assertEqual(regressor.score(None, [1,2,3]), 1.)


if __name__ == '__main__':
    unittest.main()
