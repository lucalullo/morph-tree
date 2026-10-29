"""ABI buffers stay pinned from shape validation through dispatch; zero fits."""
from array import array
import struct
import unittest
from threading import Event, Thread
from unittest.mock import patch

from morphtree._native import runtime as runtime_module
from morphtree._native.buffers import TreeCache, UnsupportedBuffers
from morphtree._native.loader import get_runtime
from morphtree._native.runtime import Runtime
from morphtree.scalable_backend import Column, ScalableBackend
from morphtree.tree import TreeNode


class NativePinning(unittest.TestCase):
    def mock_runtime(self):
        runtime = Runtime.__new__(Runtime)
        runtime.trees = TreeCache()
        runtime.traversal_function = self.unexpected_native_call
        runtime.histogram_function = self.unexpected_native_call
        return runtime

    def unexpected_native_call(self, *args):
        self.fail('Invalid preparation must fail before native dispatch')

    def test_node_numeric_callback_cannot_resize_columns(self):
        column = Column('B', [1] * 257)
        resize_blocked = []
        class Resizer:
            def __float__(self):
                try:
                    del column[:]
                except BufferError:
                    resize_blocked.append(True)
                    raise
                return 1.
        # struct.pack translates conversion failures into struct.error on
        # CPython; verify the resize itself failed, not only the outer error.
        with self.assertRaises((BufferError, struct.error)):
            self.mock_runtime().predict_tree((TreeNode(value=Resizer()),), [column])
        self.assertEqual(resize_blocked, [True])
        self.assertEqual(len(column), 257)
        column.append(2)  # Exceptional preparation releases every buffer export.

    def test_nominal_callback_cannot_invalidate_row_count(self):
        column = Column('B', [1] * 257)
        class Resizer:
            def __bool__(self):
                del column[:]
                return False
        column.nominal = Resizer()
        with self.assertRaises(BufferError):
            self.mock_runtime().predict_tree((TreeNode(value=1.),), [column])
        self.assertEqual(len(column), 257)
        column.append(2)

    def test_preparation_failure_releases_column_and_statistic_exports(self):
        column = Column('B', [1, 1])
        runtime = self.mock_runtime()
        with self.assertRaises(ValueError):
            runtime.predict_tree((TreeNode(is_leaf=None),), [column])
        column.append(1)
        values = [array('d', [1.] * 3) for _ in range(3)]
        incompatible = array('f', [1.] * 3)
        with self.assertRaises(UnsupportedBuffers):
            runtime.histogram(column, [0], incompatible, *values, 2)
        for value in (column, incompatible, *values):
            value.append(1)

    def test_concurrent_histogram_resize_is_blocked_before_shape_validation(self):
        column = Column('B', [1] * 257)
        values = [array('d', [1.] * 257) for _ in range(4)]
        ids = array('Q', [0])
        entered, attempted = Event(), Event()
        errors = []
        original = runtime_module.histogram_view
        def pause_before_validation(*args):
            entered.set()
            self.assertTrue(attempted.wait(2))
            return original(*args)
        def resize():
            if not entered.wait(2):
                errors.append('synchronization timeout')
                attempted.set()
                return
            for value in (column, ids, *values):
                try:
                    del value[:]
                except BufferError:
                    errors.append('resize blocked')
                else:
                    errors.append('unsafe resize')
            attempted.set()
        runtime = self.mock_runtime()
        calls = []
        def reject_valid_buffers(*args):
            calls.append(args[2])
            return 1  # Mocked rejection; no C function receives test pointers.
        runtime.histogram_function = reject_valid_buffers
        worker = Thread(target=resize)
        worker.start()
        try:
            with patch.object(runtime_module, 'histogram_view', pause_before_validation):
                with self.assertRaisesRegex(ValueError, 'status=1'):
                    runtime.histogram(column, ids, *values, 2)
        finally:
            entered.set()
            worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, ['resize blocked'] * 6)
        self.assertEqual(calls, [257])
        for value in (column, ids, *values):
            value.append(1)

    def test_column_container_is_snapshotted_before_node_callback(self):
        column = Column('B', [1] * 257)
        columns = [column]
        class Replacer:
            def __float__(self):
                columns[0] = Column('B', [0])
                return 1.
        runtime = self.mock_runtime()
        calls = []
        def inspect(*args):
            calls.append((args[4], args[0][0]))
            return 1
        runtime.traversal_function = inspect
        with self.assertRaisesRegex(ValueError, 'status=1'):
            runtime.predict_tree((TreeNode(value=Replacer()),), columns)
        self.assertEqual(calls, [(257, array.buffer_info(column)[0])])
        column.append(1)

    def test_valid_histograms_and_traversal_remain_bitwise_exact(self):
        runtime = Runtime(get_runtime().histogram_library)
        pure = ScalableBackend()
        bits = lambda values: b''.join(struct.pack('=d', v) for v in values)
        nodes = (TreeNode(False, feature=0, threshold_bin=1, left=1, right=2),
                 TreeNode(value=-0.), TreeNode(value=2.5))
        values = array('d', ((i % 11 - 3) * .25 for i in range(257)))
        hessians = array('d', [.2]) * 257
        for kind in ('B', 'H', 'I'):
            column = Column(kind, (i % 4 for i in range(257)), nominal=kind == 'I')
            self.assertEqual(bits(runtime.predict_tree(nodes, [column])),
                             bits(pure.predict_tree(nodes, [column])))
            args = (column, array('Q', [4, 2, 4, 0, 8] * 17),
                    values, hessians, values, hessians, 4)
            observed, copied = runtime.histogram(*args)
            self.assertEqual(copied, 0)
            self.assertEqual(b''.join(bits(v) for v in observed),
                             b''.join(bits(v) for v in pure.histogram(*args)))
        self.assertLessEqual(runtime.trees.bytes, runtime.trees.max_bytes)


if __name__ == '__main__':
    unittest.main(verbosity=2)
