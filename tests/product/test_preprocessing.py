"""TRAIN encoder contracts only: no native fit, trial, replay or LIVE evidence."""
import json
import importlib.util
from pathlib import Path
import runpy
import unittest

from morphtree._experimental.preprocessing import TrainEncoder, _m26_rows
from morphtree._experimental.target import _NamedRows, detect_target

# Missing optional packages may skip a contract; a broken installed binary must
# fail import instead of being mislabeled as an absent optional dependency.
if importlib.util.find_spec('numpy') is not None:
    import numpy as np
else:
    np = None
if importlib.util.find_spec('pandas') is not None:
    import pandas as pd
else:
    pd = None


class TrainEncoderContracts(unittest.TestCase):
    def test_numeric_missing_boolean_and_decimal_values(self):
        rows = [[1, 1.25, True], [None, float('nan'), False]]
        encoder = TrainEncoder().fit(rows)
        self.assertEqual(encoder.transform(rows), [[1.0, 1.25, 1.0], [None, None, 0.0]])

    def test_missing_and_unseen_categories_have_distinct_channels(self):
        encoder = TrainEncoder().fit([['b'], ['a'], [None]])
        self.assertEqual(encoder.transform([['a'], ['b'], [None], ['new']]),
                         [[1., 0., 0., 0.], [0., 1., 0., 0.],
                          [0., 0., 1., 0.], [0., 0., 0., 1.]])

    def test_train_only_vocabulary_and_independent_fold_encoders(self):
        parent = TrainEncoder().fit([['a'], ['b']])
        fold = TrainEncoder().fit([['a']])
        before = fold.to_dict()
        self.assertEqual(fold.transform([['b']]), [[0., 0., 1.]])
        self.assertEqual(fold.to_dict(), before)
        self.assertEqual(parent.to_dict()['columns'][0]['vocabulary'], ['a', 'b'])

    def test_named_columns_realign_and_reject_incompatible_identity(self):
        encoder = TrainEncoder().fit(_NamedRows([[1, 'a'], [2, 'b']], ['n', 'c']))
        expected = encoder.transform(_NamedRows([[3, 'new']], ['n', 'c']))
        self.assertEqual(encoder.transform(_NamedRows([['new', 3]], ['c', 'n'])), expected)
        for rows in ([[3, 'new']], _NamedRows([[3, 'new']], ['n', 'wrong']),
                     _NamedRows([[3, 'new']], ['n', 'n'])):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                encoder.transform(rows)

    def test_positional_schema_rejects_named_input(self):
        encoder = TrainEncoder().fit([[1]])
        with self.assertRaisesRegex(ValueError, 'positional'):
            encoder.transform(_NamedRows([[1]], ['x']))

    def test_schema_roundtrip_is_defensive_and_transform_stable(self):
        encoder = TrainEncoder().fit([[1, 'a'], [None, 'b']])
        state = json.loads(json.dumps(encoder.to_dict(), allow_nan=False))
        loaded = TrainEncoder.from_dict(state)
        state['columns'][1]['vocabulary'].append('untrusted')
        query = [[None, 'new'], [3, None]]
        self.assertEqual(loaded.transform(query), encoder.transform(query))
        self.assertEqual(loaded.to_dict(), encoder.to_dict())

    def test_category_and_encoded_width_caps_remain_enforced(self):
        encoder = TrainEncoder().fit([[str(i)] for i in range(64)])
        self.assertEqual(encoder.width_, 66)
        with self.assertRaisesRegex(ValueError, '64'):
            TrainEncoder().fit([[str(i)] for i in range(65)])
        with self.assertRaisesRegex(ValueError, '4096'):
            TrainEncoder.from_dict({'schema': 'morph026-train-encoder-v1', 'names': None,
                                   'columns': [{'kind': 'categorical', 'vocabulary': ['a', 'b']}] * 1025})

    def test_failed_refit_preserves_last_valid_schema(self):
        encoder = TrainEncoder().fit([[1]])
        before = encoder.to_dict()
        with self.assertRaisesRegex(ValueError, '4096'):
            encoder.fit([[str(i)] * 65 for i in range(64)])
        self.assertEqual(encoder.to_dict(), before)
        self.assertEqual(encoder.width_, 1)
        self.assertEqual(encoder.transform([[2]]), [[2.]])

    def test_incompatible_inputs_have_explicit_errors(self):
        for rows in ([], [[1], [1, 2]], [[float('inf')]], [[object()]], [['a'], [1]]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                TrainEncoder().fit(rows)
        with self.assertRaisesRegex(ValueError, 'numeric TRAIN column'):
            TrainEncoder().fit([[1]]).transform([['a']])
        with self.assertRaisesRegex(ValueError, 'string categorical column'):
            TrainEncoder().fit([['a']]).transform([[1]])

    def test_same_preprocessing_source_runs_as_standalone(self):
        # The actual standalone exporter embeds this exact source file.
        import morphtree._experimental.preprocessing as preprocessing
        namespace = runpy.run_path(str(Path(preprocessing.__file__)))
        standalone = namespace['TrainEncoder']
        encoder = TrainEncoder().fit([[1, 'a'], [None, 'b']])
        query = [[2, 'new'], [None, None]]
        self.assertEqual(standalone.from_dict(encoder.to_dict()).transform(query), encoder.transform(query))

    @unittest.skipUnless(np is not None, 'optional NumPy qualification dependency absent')
    def test_numpy_dense_inputs_and_booleans(self):
        for array in (np.array([[1, np.nan], [2, 3]]), np.array([[True], [False]])):
            with self.subTest(dtype=str(array.dtype)):
                encoder = TrainEncoder().fit(array)
                self.assertEqual(encoder.transform(array), TrainEncoder().fit(array.tolist()).transform(array.tolist()))

    @unittest.skipUnless(pd is not None, 'optional pandas qualification dependency absent')
    def test_pandas_nullable_dtypes_category_and_object(self):
        for dtype, first in [('Int64', 1), ('Float64', 1.25), ('boolean', True),
                             ('string', 'a'), ('category', 'a'), ('object', 'a')]:
            with self.subTest(dtype=dtype):
                frame = pd.DataFrame({'x': pd.Series([first, pd.NA], dtype=dtype)})
                rows, names = _m26_rows(frame)
                self.assertEqual(names, ['x'])
                self.assertEqual(rows, [[float(first) if not isinstance(first, str) else first], [None]])
                encoder = TrainEncoder().fit(frame)
                loaded = TrainEncoder.from_dict(json.loads(json.dumps(encoder.to_dict(), allow_nan=False)))
                self.assertEqual(loaded.transform(frame), encoder.transform(frame))

    @unittest.skipUnless(pd is not None, 'optional pandas qualification dependency absent')
    def test_pandas_NA_in_sequences_and_standalone_source(self):
        import morphtree._experimental.preprocessing as preprocessing
        standalone = runpy.run_path(str(Path(preprocessing.__file__)))['TrainEncoder']
        rows = [[1, 'a'], [pd.NA, pd.NA]]
        expected = [[1., 1., 0., 0.], [None, 0., 1., 0.]]
        for cls in (TrainEncoder, standalone):
            with self.subTest(cls=cls):
                self.assertEqual(cls().fit(rows).transform(rows), expected)
        with self.assertRaisesRegex(ValueError, 'Missing target'):
            detect_target([0, 1, 0, pd.NA], 'binary')


if __name__ == '__main__':
    unittest.main()
