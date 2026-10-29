"""Portable API/CLI regressions; synthetic unit weights and zero training."""
from array import array
import contextlib,importlib,io,json,runpy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from morphtree import MorphTreeClassifier as C,MorphTreeRegressor as R,scalable
from morphtree.booster import BinaryHistogramBooster
from morphtree.cli import main
from morphtree.scalable_data import Input,Table,Encoder
from morphtree.scalable_receipts import data_identity,encoder_digest

class Compatibility(unittest.TestCase):
    def model(self,task):
        fixture=importlib.import_module('test_scalable_product').ScalableProduct()
        m=fixture.model(task);m.random_state=42
        return m
    def test_none_and_auto_constructor_contract(self):
        for cls in (C,R):
            m=cls(n_jobs='auto');self.assertIsNone(m.random_state)
            self.assertEqual(m._resolved_controls()['trials'],2)
            with patch.object(m,'_fit',return_value=None),patch('random.SystemRandom.randrange',return_value=12345):m.fit([],[])
            self.assertIsNone(m.random_state);self.assertEqual(m.last_attempt_['effective_seed'],12345)
    def test_actual_native_target_buffers_reach_only_forbidden_fit(self):
        with tempfile.TemporaryDirectory() as d:
            for task in ('binary','multiclass','regression'):
                m=self.model(task);p=Path(d)/f'{task}.py';m.export_algorithm(p);program=runpy.run_path(str(p))
                res=scalable.managed_resources('auto',20);table=Table.read(Input([[float(i)] for i in range(12)]),res)
                enc=Encoder.from_state(program['SPEC']['encoder'],res)
                raw=[float(i) for i in range(12)] if task=='regression' else [i%len(m.classes_) for i in range(12)]
                resolved,targets,classes=scalable.detect_target(raw,'regression' if task=='regression' else 'classification',res.check,reserve=res.reserve)
                self.assertEqual(targets.typecode,'d' if task=='regression' else 'I')
                encoded=enc.transform(table);identity=dict(data=data_identity(table,targets,'TRAIN'),preprocessing_sha256=encoder_digest(enc))
                with patch.object(BinaryHistogramBooster,'fit',side_effect=RuntimeError('UNIT_FORBIDS_TRAINING')),self.assertRaisesRegex(RuntimeError,'UNIT_FORBIDS_TRAINING'):
                    program['train_encoded'](encoded,targets,identity=identity)
    def test_cli_all_tasks_inspect_predict_and_inference_export(self):
        with tempfile.TemporaryDirectory() as d,patch.object(BinaryHistogramBooster,'fit',side_effect=AssertionError('No training')):
            for task in ('binary','multiclass','regression'):
                m=self.model(task);p=Path(d)/f'{task}.json';m.save(p);q=Path(d)/'input.json';q.write_text('[[1.0],[null]]',encoding='utf8')
                for command in ('inspect','predict','export'):
                    result=io.StringIO();target=Path(d)/f'{task}.py'
                    with contextlib.redirect_stdout(result):self.assertEqual(main([command,'--model',str(p),'--input',str(q),'--output',str(target)]),0)
                    response=json.loads(result.getvalue())
                    if command=='predict':self.assertEqual(response['prediction'],m.predict([[1.],[None]]))
                    if command=='export':self.assertIn('def predict',target.read_text('utf8'))
    def test_cli_accepts_resource_admitted_json_beyond_two_mib(self):
        with tempfile.TemporaryDirectory() as d:
            m=self.model('binary');p=Path(d)/'model.json';m.save(p);q=Path(d)/'input.json';q.write_text('[[1.0],[2.0]]'+'\n'*(2*1024**2+32),encoding='utf8');out=io.StringIO()
            with contextlib.redirect_stdout(out):self.assertEqual(main(['predict','--model',str(p),'--input',str(q)]),0)
            self.assertEqual(len(json.loads(out.getvalue())['prediction']),2)

if __name__=='__main__':unittest.main()
