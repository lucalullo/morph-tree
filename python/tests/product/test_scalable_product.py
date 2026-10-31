"""Portable zero-fit contracts; artificial unit weights are never LIVE evidence."""
from dataclasses import asdict
import hashlib,json,math,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import morphtree
import numpy as np
from morphtree import MorphTreeClassifier as C,MorphTreeRegressor as R
from morphtree import scalable
from morphtree.config import AlgorithmGenome,HyperParams
from morphtree.booster import BinaryHistogramBooster
from morphtree.persistence import decode_booster,decode_multiclass_booster
from morphtree.scalable_data import Table,Encoder,Resources
from morphtree.scalable_backend import ScalableBackend
from morphtree.scalable_target import detect_target
from morphtree.tree import TreeNode

class ScalableProduct(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scalable.verify_source_identity()
        cls.guard=patch.object(BinaryHistogramBooster,'fit',side_effect=AssertionError('This suite forbids training'))
        cls.guard.start();cls.addClassCleanup(cls.guard.stop)
    def model(self,task='binary'):
        hp=HyperParams(n_estimators=1,max_depth=1,max_leaves=2);g=AlgorithmGenome();backend=ScalableBackend({})
        k=3 if task=='multiclass' else 1
        raw=dict(base_score=[0.]*k if k>1 else 0.,n_features=1,binner=dict(max_bins=hp.max_bins,policy='quantile',thresholds=[[0.]]),steps=[hp.learning_rate]*k,trees=[dict(nodes=[asdict(TreeNode(value=0.,depth=0))]) for _ in range(k)])
        native=decode_multiclass_booster(raw,hp,backend,k,'softmax',genome=g) if k>1 else decode_booster(raw,g,hp,backend,objective='squared_error' if task=='regression' else 'binary_logloss')
        encoder=Encoder.from_state(dict(schema='SCALABLE_TRAIN_ENCODER_1',names=None,columns=[dict(kind='numeric')]),Resources(32*1024**2,10))
        obj=R() if task=='regression' else C();obj.model_=scalable.Family(task,g,hp,encoder,[native],'boost');obj.classes_=[] if task=='regression' else list(range(3 if k>1 else 2));obj.task_=task;obj.report_=dict(scope='ARTIFICIAL_UNIT_WEIGHTS_ZERO_FITS',native_fits=0,events=[])
        return obj
    def test_defaults_preserve_evolution_and_numeric_stage(self):
        for cls in (C,R):
            m=cls(random_state=42);self.assertTrue(m.use_optuna and m.refit_full);self.assertEqual(m.max_mutations,2);self.assertEqual(m.n_trials,'auto');self.assertEqual(m._resolved_controls()['trials'],2)
            self.assertEqual(m.n_jobs,1);self.assertEqual(m.mode,'adaptive')
        self.assertTrue(hasattr(C(),'predict_proba'));self.assertFalse(hasattr(R(),'predict_proba'))
    def test_controls_identity_and_invalid_change_transaction(self):
        capacity={'max_depth':2};m=C(capacity=capacity);self.assertIs(m.get_params(deep=False)['capacity'],capacity)
        m.get_params()['capacity']['max_depth']=9;self.assertEqual(capacity,{'max_depth':2})
        before=m.__dict__.copy()
        with self.assertRaises(ValueError):m.set_params(n_jobs=2)
        self.assertEqual(m.__dict__,before);self.assertIs(m.set_params(),m)
        self.assertEqual(C(n_trials=7,max_mutations=10)._resolved_controls()['mutations'],3)
    def test_real_resource_admission_precedes_native_or_feature_copy(self):
        import numpy as np
        m=C(n_trials=1000000,memory_budget=1024**2,time_budget=5,checkpoint_dir=None)
        with patch.object(Table,'read',side_effect=AssertionError('Early feature copy')),self.assertRaises(MemoryError):m.fit(np.zeros((100,2)),np.arange(100)%2)
        self.assertEqual(m.last_attempt_['native_fits'],0)
    def test_high_class_cardinality_uses_resource_budget(self):
        y=list(range(129))*4;res=Resources(32*1024**2,10)
        task,values,classes=detect_target(y,'classification',res.check,reserve=res.reserve)
        self.assertEqual(task,'multiclass');self.assertEqual(classes,list(range(129)));self.assertEqual(list(values),y)
    def test_numpy_nan_high_cardinality_and_encoder_roundtrip(self):
        import numpy as np
        class NoLists(np.ndarray):
            def tolist(self):raise AssertionError('No full-input tolist')
        rows=np.array([[float('nan') if i%7==0 else float(i),f'category-{i}'] for i in range(257)],dtype=object).view(NoLists)
        res=Resources(64*1024**2,10);table=Table.read(rows,res);enc=Encoder().fit(table)
        self.assertEqual(len(table),257);self.assertEqual(len(enc.columns[1]['vocabulary']),257)
        restored=Encoder.from_state(json.loads(json.dumps(enc.state())),res);self.assertEqual(enc.state(),restored.state())
        query=Table.read([[None,'unseen'],[2.,'category-2']],res)
        for left,right in zip(enc.transform(query),restored.transform(query)):
            for x,y in zip(left,right):
                if math.isnan(x):self.assertTrue(math.isnan(y))
                else:self.assertEqual(x,y)
    def test_all_task_artificial_model_io_and_generic_loader(self):
        with tempfile.TemporaryDirectory() as d:
            for task in ('binary','multiclass','regression'):
                m=self.model(task);path=Path(d)/(task+'.json');h=m.save(path);loaded=morphtree.load_model(path,sha256=h)
                self.assertEqual(loaded.model_.packet(),m.model_.packet());np.testing.assert_array_equal(loaded.predict([[1.],[None]]),m.predict([[1.],[None]]))
                self.assertEqual(set(loaded.available_variants()),{'selected'})
                self.assertTrue(loaded.get_algorithm_program()['training_program_export'])
                if task!='regression':np.testing.assert_array_equal(loaded.predict_proba([[1.]]),m.predict_proba([[1.]]))
    def test_standalone_is_stdlib_only_and_prediction_exact(self):
        with tempfile.TemporaryDirectory() as d:
            m=self.model();path=Path(d)/'predict.py';m.export_standalone(path)
            code='import json,runpy,sys;m=runpy.run_path(sys.argv[1]);print(json.dumps(m["predict_proba"]([[1.],[None]])))'
            r=subprocess.run([sys.executable,'-I','-B','-S','-c',code,str(path)],capture_output=True,text=True,timeout=10)
            self.assertEqual(r.returncode,0,r.stderr);np.testing.assert_array_equal(json.loads(r.stdout),m.predict_proba([[1.],[None]]))
    def test_invalid_json_and_changed_hash_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.json';m=self.model();h=m.save(p);p.write_bytes(p.read_bytes()+b' ')
            with self.assertRaises(ValueError):C.load(p,sha256=h)
            for raw in ('{}','{"schema":"unknown"}','{"schema":NaN}'):
                p.write_text(raw,encoding='utf8')
                with self.assertRaises(ValueError):C.load(p)
    def test_absent_variants_never_create_weights_or_write_files(self):
        m=self.model()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'absent.json'
            with self.assertRaises(ValueError):m.save(p,variant='tuned')
            self.assertFalse(p.exists())
    def test_failed_refit_preserves_model_and_successful_parameter_update_resets(self):
        m=self.model();before=m.model_
        with patch.object(m,'_fit',side_effect=ValueError('Unit boundary,not a native fit')),self.assertRaises(ValueError):m.fit([],[])
        self.assertIs(m.model_,before);self.assertEqual(m.last_attempt_['native_fits'],0)
        m.set_params(random_state=7);self.assertFalse(hasattr(m,'model_'))
    def test_source_identity_matches_installed_manifest(self):
        package=Path(morphtree.__file__).resolve().parent;manifest=json.loads((package/'_experimental/manifest.json').read_text('utf8'))
        # Native Lab extends custody to packaged C sources,binary and ABI
        # metadata. The manifest itself is excluded from its own hash map.
        own=package/'_experimental/manifest.json'
        self.assertEqual({p.relative_to(package).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in package.rglob('*') if p.is_file() and p!=own and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')},manifest['files'])

if __name__=='__main__':unittest.main(verbosity=2)
