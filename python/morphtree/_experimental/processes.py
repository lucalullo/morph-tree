import os,stat,time,signal,subprocess
from pathlib import Path
from contextlib import contextmanager
from .runtime_io import artifact_path
class ContinuityError(ValueError):
    """Invalid plan, changed evidence, or an unsafe recovery request."""

class LockBusy(ContinuityError):
    """Another live supervisor owns this state directory."""

def _safe(path):
    path = Path(os.path.abspath(path))
    for item in reversed((path, *path.parents)):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 1024:
            raise ContinuityError('Symlinks and reparse points are forbidden')
    return path

def _artifact(path):
    try:
        return artifact_path(path)
    except ValueError as exc:
        raise ContinuityError(str(exc)) from exc

@contextmanager
def exclusive_lock(state_dir):
    """Advisory kernel lock; closure or process death releases ownership."""
    root = _artifact(state_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = _safe(root / '.lock')
    with path.open('a+b') as stream:
        if path.stat().st_size == 0:
            stream.write(b'\x00')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise LockBusy('Another live supervisor owns the state directory') from exc
        try:
            yield root
        finally:
            stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

class ChildGuard:
    """Windows job kills descendants on parent death; POSIX uses a process group."""

    def __init__(self, child, *, memory_limits=None):
        self.child = child
        self.job = None
        self._owned_pids = {child.pid}
        self._working_set_peaks = {}
        self.memory_limits = memory_limits
        if memory_limits is not None:
            if os.name != 'nt' or type(memory_limits) is not dict or set(memory_limits) != {'process_commit_bytes', 'job_commit_bytes'} or any((type(v) is not int or v < 32 * 1024 ** 2 for v in memory_limits.values())) or (memory_limits['process_commit_bytes'] > memory_limits['job_commit_bytes']):
                if child.poll() is None:
                    child.kill()
                child.wait()
                raise ContinuityError('Windows private-commit memory profile unavailable/invalid; not an RSS limit')
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes

            class Basic(ctypes.Structure):
                _fields_ = [('PerProcessUserTimeLimit', ctypes.c_int64), ('PerJobUserTimeLimit', ctypes.c_int64), ('LimitFlags', wintypes.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t), ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', wintypes.DWORD), ('Affinity', ctypes.c_size_t), ('PriorityClass', wintypes.DWORD), ('SchedulingClass', wintypes.DWORD)]

            class IO(ctypes.Structure):
                _fields_ = [(name, ctypes.c_uint64) for name in ('ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount', 'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]

            class Extended(ctypes.Structure):
                _fields_ = [('BasicLimitInformation', Basic), ('IoInfo', IO), ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t), ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]

            class Accounting(ctypes.Structure):
                _fields_ = [(name, ctypes.c_int64) for name in ('TotalUserTime', 'TotalKernelTime', 'ThisPeriodTotalUserTime', 'ThisPeriodTotalKernelTime')] + [(name, wintypes.DWORD) for name in ('TotalPageFaultCount', 'TotalProcesses', 'ActiveProcesses', 'TotalTerminatedProcesses')]
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
            kernel.CreateJobObjectW.restype = wintypes.HANDLE
            kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
            kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
            kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
            kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            self.kernel = kernel
            self.accounting = Accounting
            self.extended = Extended
            self.job = kernel.CreateJobObjectW(None, None)
            limit = Extended()
            limit.BasicLimitInformation.LimitFlags = 8192
            if memory_limits is not None:
                limit.BasicLimitInformation.LimitFlags |= 256 | 512
                limit.ProcessMemoryLimit = memory_limits['process_commit_bytes']
                limit.JobMemoryLimit = memory_limits['job_commit_bytes']
            if not self.job or not kernel.SetInformationJobObject(self.job, 9, ctypes.byref(limit), ctypes.sizeof(limit)) or (not kernel.AssignProcessToJobObject(self.job, int(child._handle))):
                if self.job:
                    kernel.CloseHandle(self.job)
                    self.job = None
                child.kill()
                child.wait()
                raise ContinuityError('Cannot establish Windows process-tree cleanup job')

    def statistics(self):
        """Owned Job accounting only. Commit peaks are NOT RSS/working-set peaks."""
        if self.job is None:
            return {'scope': 'UNAVAILABLE', 'job_private_commit_peak_bytes': None, 'CPU_job_seconds': None}
        import ctypes
        limit = self.extended()
        account = self.accounting()
        if not self.kernel.QueryInformationJobObject(self.job, 9, ctypes.byref(limit), ctypes.sizeof(limit), None) or not self.kernel.QueryInformationJobObject(self.job, 1, ctypes.byref(account), ctypes.sizeof(account), None):
            raise ContinuityError('Cannot verify owned Job limits/accounting')
        actual = {'process_commit_bytes': int(limit.ProcessMemoryLimit), 'job_commit_bytes': int(limit.JobMemoryLimit)}
        flags = int(limit.BasicLimitInformation.LimitFlags)
        if self.memory_limits is not None and (actual != self.memory_limits or flags & 768 != 768):
            raise ContinuityError('Owned Job memory limits differ')
        from ctypes import wintypes

        class Memory(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage')]
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        for pid in self._owned_pids:
            handle = self.kernel.OpenProcess(1024 | 16, False, pid)
            if not handle:
                continue
            try:
                member = wintypes.BOOL()
                if not self.kernel.IsProcessInJob(handle, self.job, ctypes.byref(member)) or not member.value:
                    continue
                memory = Memory()
                memory.cb = ctypes.sizeof(memory)
                if psapi.GetProcessMemoryInfo(handle, ctypes.byref(memory), ctypes.sizeof(memory)):
                    self._working_set_peaks[pid] = max(self._working_set_peaks.get(pid, 0), int(memory.PeakWorkingSetSize))
            finally:
                self.kernel.CloseHandle(handle)
        return {'scope': 'WINDOWS_OWNED_JOB_PRIVATE_COMMIT_NOT_RSS', 'memory_limits': actual, 'private_commit_allocation_policy_verified': self.memory_limits is not None, 'hard_private_commit_cap_verified': False, 'hard_peak_private_commit_cap': False, 'peak_over_job_policy_bytes': max(0, int(limit.PeakJobMemoryUsed) - actual['job_commit_bytes']) if self.memory_limits else None, 'memory_boundary': 'Configured Windows allocation policy;observed refusal is not proof of an absolute peak ceiling', 'job_private_commit_peak_bytes': int(limit.PeakJobMemoryUsed), 'process_private_commit_peak_bytes': int(limit.PeakProcessMemoryUsed), 'CPU_job_seconds': (account.TotalUserTime + account.TotalKernelTime) / 10000000.0, 'active_processes': int(account.ActiveProcesses), 'total_processes': int(account.TotalProcesses), 'owned_process_peak_working_sets_bytes': dict(self._working_set_peaks), 'working_set_scope': 'Sampled OS-maintained peaks of verified owned launcher/worker PIDs;not aggregate job RSS', 'peak_RSS': 'NOT_A_PORTABLE_OR_AGGREGATE_RSS_MEASURE', 'aggregate_RSS_cap': False}

    def include_pid(self, pid):
        """Also attach an interpreter already started by a Windows venv launcher."""
        if self.job is None:
            return
        import ctypes
        from ctypes import wintypes
        handle = self.kernel.OpenProcess(256 | 1 | 4096, False, pid)
        if not handle:
            raise ContinuityError('Cannot identify the bootstrap interpreter')
        try:
            member = wintypes.BOOL()
            if not self.kernel.IsProcessInJob(handle, self.job, ctypes.byref(member)):
                raise ContinuityError('Cannot verify bootstrap job membership')
            if not member.value and (not self.kernel.AssignProcessToJobObject(self.job, handle)):
                raise ContinuityError('Cannot attach bootstrap interpreter to cleanup job')
            self._owned_pids.add(pid)
        finally:
            self.kernel.CloseHandle(handle)

    def close(self):
        if self.job is not None:
            import ctypes
            self.kernel.TerminateJobObject(self.job, 1)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                counters = self.accounting()
                if not self.kernel.QueryInformationJobObject(self.job, 1, ctypes.byref(counters), ctypes.sizeof(counters), None) or counters.ActiveProcesses == 0:
                    break
                time.sleep(0.01)
            self.kernel.CloseHandle(self.job)
            self.job = None
        elif os.name != 'nt':
            try:
                os.killpg(self.child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if self.child.poll() is None:
            self.child.kill()
        self.child.wait()
