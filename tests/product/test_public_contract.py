"""Default installed product suite. No actual native fit or numerical trial."""
import ast,importlib.metadata,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import morphtree,optuna
from morphtree import MorphTree,booster,tree

class PublicContract(unittest.TestCase):
 def setUp(self):
  self.patches=[]
  def forbidden(*a,**k):raise AssertionError('NO_NATIVE_FIT_OR_OPTUNA_TRIAL')
  for name in ('BinaryHistogramBooster','MulticlassHistogramBooster','RegressionHistogramBooster'):
   typ=getattr(booster,name,None)
   if typ:self.patches.append(patch.object(typ,'fit',forbidden))
  self.patches += [patch.object(tree.HistogramTree,'fit',forbidden),patch.object(optuna,'create_study',forbidden)]
  for p in self.patches:p.start()
  self.addCleanup(lambda:[p.stop() for p in reversed(self.patches)])
 def test_canonical_defaults_three_tasks(self):
  for task in ('binary','multiclass','regression'):
   m=MorphTree(task=task,random_state=42,verbosity=0)
   self.assertTrue(m._evolving_program);self.assertEqual(m.tuning,'optuna');self.assertEqual(m._max_mutations,2)
 def test_disable_only_numerical_tuning(self):
  m=MorphTree(use_optuna=False);self.assertTrue(m._evolving_program);self.assertEqual(m.tuning,'none')
 def test_distribution_and_native_format_are_distinct(self):
  self.assertEqual(morphtree.__version__,'1.0.0')
  self.assertEqual(importlib.metadata.version('morphtree'),'1.0.0')
  self.assertEqual(morphtree.__native_core_version__,'0.6.0.dev0')
 def test_invalid_controls_are_rejected(self):
  for kwargs in ({'max_mutations':-1},{'use_optuna':'yes'},{'verbosity':7},{'n_jobs':2}):
   with self.subTest(kwargs=kwargs):
    with self.assertRaises((ValueError,TypeError)):MorphTree(**kwargs)
 def test_malformed_model_never_becomes_training(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder)
   for i,data in enumerate((b'[]',b'{"x":NaN}',b'{"x":1,"x":2}',b'{"schema":"UNKNOWN"}')):
    path=root/(str(i)+'.json');path.write_bytes(data)
    with self.assertRaises((ValueError,TypeError)):MorphTree.load(path)
 def test_untrusted_nested_model_objects_are_rejected(self):
  from morphtree._experimental.base import _load_members
  for task in ('binary','multiclass','regression'):
   with self.assertRaises(ValueError):_load_members(task,[None])
  body={'schema':'product006-native-control-v1',
    'constructor':MorphTree(use_optuna=False).get_params(),
    'selected_variant':'pure','variants':{'pure':None},
    'origins':{'pure':{'origin':'PRE_FINAL_NUMERIC_SEARCH','evidence':'NEGATIVE_SCHEMA_FIXTURE_NOT_TRAINED'}}}
  with tempfile.TemporaryDirectory() as folder:
   p=Path(folder)/'bad.json';p.write_text(json.dumps(body),encoding='utf8')
   with self.assertRaises(ValueError):MorphTree.load(p)
 def test_runtime_has_no_vendor_learner_import(self):
  package=Path(morphtree.__file__).parent
  for path in package.rglob('*.py'):
   for node in ast.walk(ast.parse(path.read_text('utf8'))):
    names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
    self.assertFalse(any(n.split('.')[0] in ('xgboost','lightgbm','catboost','sklearn') for n in names))

if __name__=='__main__':unittest.main()
