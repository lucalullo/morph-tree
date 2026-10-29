"""Observed shared-runtime concurrency contract,no fitting or algorithm changes."""
import struct,sys,unittest
from concurrent.futures import ThreadPoolExecutor
from array import array
from morphtree._native.loader import get_runtime
from morphtree.scalable_backend import ScalableBackend,Column
from morphtree.tree import TreeNode
class InstalledThreads(unittest.TestCase):
    def test_shared_runtime_histogram_and_traversal_remain_exact(self):
        runtime=get_runtime();col=Column('B',(i%4 for i in range(2003)));v=array('d',((i%11-3)*.25 for i in range(len(col))));h=array('d',[.2])*len(col)
        nodes=(TreeNode(False,feature=0,threshold_bin=1,left=1,right=2,missing_left=True),TreeNode(value=-0.),TreeNode(value=2.5));pure=ScalableBackend();args=(col,[4,2,4,0,8]*17,v,h,v,h,4)
        bits=lambda a:b''.join(struct.pack('=d',x) for x in a)
        expected=bits(pure.predict_tree(nodes,[col]));hist=b''.join(bits(r) for r in pure.histogram(*args));old=sys.getswitchinterval();sys.setswitchinterval(1e-6)
        def query(i):
            n=nodes if i%2==0 else (nodes[0],TreeNode(value=0.),nodes[2]);target=expected if i%2==0 else bits(pure.predict_tree(n,[col]));self.assertEqual(bits(runtime.predict_tree(n,[col])),target);self.assertEqual(b''.join(bits(r) for r in runtime.histogram(*args)[0]),hist);return True
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:self.assertTrue(all(pool.map(query,range(48))))
        finally:sys.setswitchinterval(old)
        self.assertLessEqual(runtime.trees.bytes,runtime.trees.max_bytes)
        # Test outcomes belong to the runner's report, not a persistent file in
        # the caller's working directory. Repeated runs exercise the same cache.
