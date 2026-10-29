"""Installed-capable integrated engine contracts,no model/tree fitting."""
from array import array
import ast,ctypes,hashlib,json,struct,tempfile,unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from morphtree._native import loader
from morphtree.native_lab_backend import NativeBackend,bind_backend,python_backend_scope,native_backend_scope
from morphtree.scalable_backend import ScalableBackend,Column
from morphtree.tree import TreeNode
from morphtree.scalable import verify_source_identity

def fixture(rows):
    nodes=(TreeNode(False,feature=0,threshold_bin=1,left=1,right=2,missing_left=True),TreeNode(value=-0.),TreeNode(value=2.5))
    return nodes,[Column('B',(i%4 for i in range(rows)))]
def bits(values):return b''.join(struct.pack('=d',v) for v in values)

class IntegratedContracts(unittest.TestCase):
    def test_default_provider_and_verified_package_binary(self):
        verify_source_identity();backend=bind_backend()({});self.assertIsInstance(backend,NativeBackend)
        p,manifest=loader.artifact();self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),manifest['binary_sha256']);self.assertFalse(manifest['runtime_compilation'])
        self.assertIsInstance(backend.session.kernel.histogram_library,ctypes.PyDLL)
    def test_reference_scope_is_explicit_and_restored(self):
        with python_backend_scope():self.assertIs(bind_backend(),ScalableBackend)
        self.assertIsInstance(bind_backend()({}),NativeBackend)
        with native_backend_scope():
            with self.assertRaises(RuntimeError):
                with python_backend_scope():pass
    def test_small_batch_native_route_and_all_rows_exact(self):
        with native_backend_scope() as session:
            backend=bind_backend()({})
            for rows in (1,8,32,64,127,128,512,2003):
                n,c=fixture(rows);self.assertEqual(bits(backend.predict_tree(n,c)),bits(ScalableBackend().predict_tree(n,c)))
            self.assertEqual(session.small_batch_python_calls,5);self.assertEqual(session.traversal_calls,3);self.assertEqual(session.traversal_fallback_calls,0)
    def test_nominal_routes_use_equality_and_unknown_codes(self):
        n,c=fixture(257);c=[Column('I',(i%7 for i in range(257)),nominal=True)]
        self.assertEqual(bind_backend()({0:4}).predict_tree(n,c),ScalableBackend({0:4}).predict_tree(n,c))
    def test_histogram_all_statistics_are_bitwise_unchanged(self):
        b=bind_backend()({});n=257;ids=[4,2,4,0,8]*17;v=array('d',((i%11-3)*.25 for i in range(n)));h=array('d',[.2])*n
        for kind in ('B','H','I'):
            c=Column(kind,(i%17 for i in range(n)),nominal=kind=='I');args=(c,ids,v,h,v,h,17)
            a=b.histogram(*args);ref=ScalableBackend().histogram(*args)
            self.assertEqual(b''.join(bits(row) for row in a),b''.join(bits(row) for row in ref));self.assertEqual(a.nominal,kind=='I')
        self.assertEqual(b.session.calls,3);self.assertEqual(b.session.fallback_calls,0)
    def test_native_rejection_does_not_hide_behind_fallback(self):
        b=bind_backend()({});c=Column('B',[1,1]);v=array('d',[1.,2.])
        with self.assertRaises(ValueError):b.histogram(c,array('Q',[2]),v,v,v,v,2)
        self.assertEqual(b.session.fallback_calls,0)
    def test_missing_and_corrupted_binary_refused_before_loading(self):
        package=Path(loader.__file__).parent;manifest=json.loads((package/'manifest.json').read_text('utf8'))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with patch.object(loader,'__file__',str(root/'loader.py')):
                with self.assertRaises(ImportError):loader.artifact()
                (root/'manifest.json').write_text(json.dumps(manifest),encoding='utf8')
                binary=root/manifest['library'];binary.parent.mkdir(parents=True);binary.write_bytes(b'not a DLL')
                with self.assertRaisesRegex(ImportError,'integrity'):loader.artifact()
    def test_manifest_cannot_select_an_external_binary(self):
        manifest=json.loads((Path(loader.__file__).parent/'manifest.json').read_text('utf8'))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);manifest['library']='../foreign.dll';(root/'manifest.json').write_text(json.dumps(manifest),encoding='utf8')
            with patch.object(loader,'__file__',str(root/'loader.py')),self.assertRaises(ImportError):loader.artifact()
    def test_cached_mutation_and_signed_zero_are_not_stale(self):
        runtime=loader.get_runtime();n,c=fixture(257);runtime.predict_tree(n,c);changed=(n[0],replace(n[1],value=0.),n[2])
        self.assertEqual(bits(runtime.predict_tree(changed,c)),bits(ScalableBackend().predict_tree(changed,c)))
        object.__setattr__(changed[1],'value',7.);self.assertEqual(runtime.predict_tree(changed,c),ScalableBackend().predict_tree(changed,c))
        self.assertLessEqual(runtime.trees.bytes,runtime.trees.max_bytes)
    def test_instance_buffer_pointer_override_is_ignored(self):
        c=Column('B',[1,1]);foreign=array('B',[0,0]);c.buffer_info=lambda:array.buffer_info(foreign)
        v=array('d',[1.,1.]);args=(c,array('Q',[0,1]),v,v,v,v,2);b=bind_backend()({})
        self.assertEqual(b.histogram(*args),ScalableBackend().histogram(*args))
        nodes=(TreeNode(False,feature=0,threshold_bin=0,left=1,right=2),TreeNode(value=0.),TreeNode(value=1.))
        self.assertEqual(loader.get_runtime().predict_tree(nodes,[c]),[1.,1.])
    def test_array_subclass_cannot_spoof_a_native_pointer_or_size(self):
        from morphtree._native.buffers import columns_view,histogram_view,UnsupportedBuffers
        class Liar(array):
            def __len__(self):return 1000000
            def buffer_info(self):raise AssertionError('Must never call overridden pointer method')
        c=Liar('B',[1,1]);v=array('d',[1.,1.])
        with self.assertRaises(UnsupportedBuffers):columns_view([c])
        with self.assertRaises(UnsupportedBuffers):histogram_view(c,[0],v,v,v,v,2)
    def test_node_container_cannot_spoof_packed_array_length(self):
        from morphtree._native.buffers import UnsupportedBuffers
        nodes,columns=fixture(257)
        class Short(list):
            def __len__(self):return 1
        class Large(tuple):
            def __len__(self):return 1000000
        for custom in (Short(nodes),Large(nodes)):
            with self.assertRaises(UnsupportedBuffers):loader.get_runtime().predict_tree(custom,columns)
        # The public adapter falls back safely for an unsupported container;
        # no misleading shape is ever forwarded to the C ABI.
        backend=bind_backend()({})
        self.assertEqual(backend.predict_tree(Short(nodes),columns),ScalableBackend().predict_tree(Short(nodes),columns))
        self.assertEqual(backend.session.traversal_fallback_calls,1)
    def test_subnormal_cancellation_and_nonfinite_histogram_semantics(self):
        import math
        c=Column('B',[0,0,0,1,1]);ids=array('Q',range(5));tiny=math.nextafter(0.,1.)
        g=array('d',[1e16,1.,-1e16,tiny,tiny]);h=array('d',[1.]*5);args=(c,ids,g,h,g,h,2)
        a=bind_backend()({}).histogram(*args);b=ScalableBackend().histogram(*args)
        self.assertEqual(b''.join(bits(r) for r in a),b''.join(bits(r) for r in b))

if __name__=='__main__':unittest.main(verbosity=2)
