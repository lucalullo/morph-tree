"""Protect native dispatch while the public evolutionary API is migrated."""
import importlib
import unittest
from unittest.mock import patch

import morphtree
from morphtree import model
from morphtree._experimental import training


class NativeDispatchContracts(unittest.TestCase):
    def test_public_facade_names_cannot_redirect_native_dispatch(self):
        expected = {'binary': model.MorphTreeClassifier,
                    'multiclass': model.MorphTreeMulticlassClassifier,
                    'regression': model.MorphTreeRegressor}
        try:
            with patch.multiple(morphtree, MorphTreeClassifier=object(),
                                MorphTreeMulticlassClassifier=object(),
                                MorphTreeRegressor=object()):
                importlib.reload(training)
                self.assertEqual(training.NATIVE, expected)
        finally:
            importlib.reload(training)


if __name__ == '__main__':
    unittest.main()
