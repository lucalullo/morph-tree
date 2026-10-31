"""Public output contracts on artificial unit weights; never fit a learner."""
import importlib.util,json,runpy,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import morphtree
import test_scalable_product as fixtures
from morphtree.prediction_output import prediction_array,probability_array
from morphtree.native_lab_backend import python_backend_scope

class PredictionArrays(unittest.TestCase):
    def model(self,task='binary'):
        m=fixtures.ScalableProduct().model(task)
        m.model_.models[0].base_score=[-.7,.2,.5] if task=='multiclass' else .37
        return m
    def test_classification_shape_dtype_slicing_and_class_order(self):
        for task,classes in [('binary',['GALAXY','STAR']),('multiclass',['GALAXY','QSO','STAR'])]:
            m=self.model(task);m.classes_=classes
            for n in (1,129,17003):
                X=[[1.],[None]]*(n//2)+([[1.]] if n%2 else [])
                pred=m.predict(X);p=m.predict_proba(X)
                self.assertIsInstance(pred,np.ndarray);self.assertEqual(pred.shape,(n,))
                self.assertIsInstance(p,np.ndarray);self.assertEqual(p.shape,(len(X),len(m.classes_)))
                self.assertEqual(p.dtype,np.dtype('float64'))
                self.assertTrue(np.isfinite(p).all());self.assertTrue(((p>=0)&(p<=1)).all())
                np.testing.assert_allclose(p.sum(axis=1),1.,rtol=0,atol=1e-15)
                np.testing.assert_array_equal(pred,np.asarray(classes,dtype=object)[p.argmax(axis=1)])
                self.assertEqual(p[:,1].shape,(n,));self.assertEqual((p[:,1]>.1).dtype,np.dtype('bool'))
    def test_regression_is_numeric_one_dimensional_and_hides_proba(self):
        m=self.model('regression')
        for X in ([[1.]],[[1.],[None]]*9000):
            p=m.predict(X);self.assertIsInstance(p,np.ndarray);self.assertEqual(p.shape,(len(X),));self.assertEqual(p.dtype,np.dtype('float64'))
            np.testing.assert_array_equal(p,np.full(len(X),.37))
        self.assertFalse(hasattr(m,'predict_proba'))
    def test_labels_do_not_lose_large_integer_identity(self):
        for classes in ([0,2**53+1],[-1,2**64-1],[0,2**80],['A','STAR']):
            m=self.model();m.classes_=list(classes)
            self.assertEqual(m.predict([[1.]])[0].item() if hasattr(m.predict([[1.]])[0],'item') else m.predict([[1.]])[0],classes[1])
            self.assertEqual(prediction_array(classes).tolist(),list(classes))
    def test_invalid_probabilities_rejected_without_clipping_or_renormalizing(self):
        for bad in ([[np.nan,.5]],[[np.inf,0.]],[[-.1,1.1]],[[.3,.3]],[[1.]],[[[.5,.5]]]):
            with self.assertRaises(ValueError):probability_array(bad,2,n_samples=1)
        good=[[.1234567890123456,1-.1234567890123456]]
        np.testing.assert_array_equal(probability_array(good,2),np.asarray(good))
    def test_python_and_native_load_paths_are_bitwise_equal(self):
        with tempfile.TemporaryDirectory() as d:
            for task in ('binary','multiclass','regression'):
                m=self.model(task);path=Path(d)/(task+'.json');m.save(path)
                with python_backend_scope():pure=morphtree.load_model(path)
                native=morphtree.load_model(path);X=[[1.],[None]]*257
                np.testing.assert_array_equal(native.predict(X),pure.predict(X))
                if task!='regression':np.testing.assert_array_equal(native.predict_proba(X),pure.predict_proba(X))
                self.assertEqual(native.model_.packet(),pure.model_.packet())
    def test_save_load_variants_and_failed_refit_preserve_output(self):
        from morphtree.scalable import _Scalable
        with tempfile.TemporaryDirectory() as d:
            for task in ('binary','multiclass','regression'):
                m=self.model(task);X=[[1.],[None]];before=m.predict(X);packet=m.model_.packet();history=m.get_mutation_history()
                path=Path(d)/(task+'.json');h=m.save(path);loaded=morphtree.load_model(path,sha256=h)
                np.testing.assert_array_equal(loaded.predict(X),before)
                np.testing.assert_array_equal(loaded._variant_view('selected').predict(X),before)
                with patch.object(_Scalable,'_fit',side_effect=RuntimeError('NO_FIT')),self.assertRaisesRegex(RuntimeError,'NO_FIT'):loaded.fit([],[])
                np.testing.assert_array_equal(loaded.predict(X),before);self.assertEqual(loaded.model_.packet(),packet);self.assertEqual(loaded.get_mutation_history(),history)
    def test_numpy_export_and_stdlib_export_have_explicit_formats(self):
        with tempfile.TemporaryDirectory() as d:
            for task in ('binary','multiclass','regression'):
                m=self.model(task);X=[[1.],[None]]
                p=Path(d)/(task+'-numpy.py');m.export_standalone(p,output_format='numpy');exported=runpy.run_path(str(p))
                self.assertIsInstance(exported['predict'](X),np.ndarray);np.testing.assert_array_equal(exported['predict'](X),m.predict(X))
                if task!='regression':
                    a=exported['predict_proba'](X);self.assertIsInstance(a,np.ndarray);np.testing.assert_array_equal(a,m.predict_proba(X));self.assertEqual(a[:,1].shape,(2,))
                plain=Path(d)/(task+'-python.py');m.export_standalone(plain)
                code='import json,runpy,sys;m=runpy.run_path(sys.argv[1]);v=m["predict"]([[1.],[None]]);assert isinstance(v,list);print(json.dumps(v))'
                child=subprocess.run([sys.executable,'-I','-B','-S','-c',code,str(plain)],capture_output=True,text=True,timeout=10)
                self.assertEqual(child.returncode,0,child.stderr);np.testing.assert_array_equal(json.loads(child.stdout),m.predict(X))
    def test_sklearn_view_does_not_reintroduce_list_or_truth_value_errors(self):
        for task in ('binary','multiclass','regression'):
            m=self.model(task);X=[[1.],[None]];v=m.as_sklearn()
            np.testing.assert_array_equal(v.predict(X),m.predict(X))
            if task!='regression':np.testing.assert_array_equal(v.predict_proba(X),m.predict_proba(X))
    def test_score_remains_python_float(self):
        for task in ('binary','multiclass','regression'):
            m=self.model(task);y=[.37,.38] if task=='regression' else [0,1]
            self.assertIs(type(m.score([[1.],[None]],y)),float)
        m=self.model();m.classes_=['A','STAR']
        self.assertIs(type(m.score([[1.],[None]],['A','STAR'])),float)

if __name__=='__main__':unittest.main()
