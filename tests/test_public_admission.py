"""Scalable public dispatch and metadata admission; no learner or tree fits."""
from array import array
import unittest
from unittest.mock import patch
import morphtree
from morphtree import scalable
from morphtree.scalable_data import Resources,Table

class PublicAdmission(unittest.TestCase):
    def test_public_names_select_the_scalable_implementation(self):
        self.assertIs(morphtree.MorphTreeClassifier,scalable.MorphTreeClassifier)
        self.assertIs(morphtree.MorphTreeRegressor,scalable.MorphTreeRegressor)

    def test_large_shape_reaches_column_reader_without_legacy_caps(self):
        # Metadata only: deliberately no feature values or column allocation.
        # Controlled headroom makes this a deterministic admission test, not a
        # claim of actual hardware capacity or successful large-data training.
        class ShapeOnly:
            shape=(100003,129)
        class BeforeColumns(Exception):pass
        def resources():
            value=Resources(2*1024**3,30)
            value.available_bytes_snapshot=4*1024**3
            value.memory_observation={'scope':'DETERMINISTIC_TEST_HEADROOM_NOT_OBSERVED_RAM'}
            return value
        for task in ('binary','multiclass','regression'):
            y=array('d',(i/100003 for i in range(100003))) if task=='regression' else array('I',(i%(3 if task=='multiclass' else 2) for i in range(100003)))
            cls=morphtree.MorphTreeRegressor if task=='regression' else morphtree.MorphTreeClassifier
            model=cls(random_state=42,checkpoint_dir=None)
            with self.subTest(task=task),patch.object(model,'_resources',side_effect=resources),patch.object(Table,'read',side_effect=BeforeColumns) as read,patch('morphtree.api._new_user_session',side_effect=AssertionError('Legacy admission must not run')):
                with self.assertRaises(BeforeColumns):model.fit(ShapeOnly(),y)
                self.assertEqual(read.call_count,1)
                plan=model.last_attempt_['resource_plan']
                self.assertEqual((plan['rows'],plan['features']),(100003,129))
                self.assertEqual(model.last_attempt_['native_fits'],0)

if __name__=='__main__':unittest.main()
