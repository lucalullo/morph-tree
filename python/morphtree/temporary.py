"""One-attempt private temporary directories and ownership-checked cleanup.

No global tempfile configuration or retry loop is changed. Names have the same
eight-character random suffix length as the previous stdlib directories, using
48 fresh cryptographic bits per name. Creation is exclusive; a collision or
permission failure propagates without retry. These are OS resource helpers,
not learner code or a guarantee against arbitrary filesystem latency.
"""
import os
from pathlib import Path
import secrets
import shutil
import stat
import tempfile
import weakref


def directory(*,prefix='tmp',dir=None):
    if type(prefix) is not str or not prefix or any(c in prefix for c in ('/','\\','\0',':')) or Path(prefix).name!=prefix or prefix in ('.','..'):
        raise ValueError('A single temporary-name prefix is required')
    parent=Path(tempfile.gettempdir() if dir is None else dir).resolve(strict=True)
    if not parent.is_dir():raise NotADirectoryError(parent)
    path=parent/(prefix+secrets.token_urlsafe(6))
    os.mkdir(path,0o700)
    return path


def _remove_owned(name,parent,identity):
    path=Path(name);parent=Path(parent)
    try:current=path.lstat()
    except FileNotFoundError:return
    if stat.S_ISLNK(current.st_mode) or getattr(path,'is_junction',lambda:False)():
        raise ValueError('Owned temporary cleanup refuses a link')
    if not stat.S_ISDIR(current.st_mode) or (current.st_dev,current.st_ino)!=identity:
        raise ValueError('Owned temporary directory identity changed')
    if path.parent.resolve()!=parent or not path.resolve().is_relative_to(parent):
        raise ValueError('Owned temporary cleanup escaped its original parent')
    shutil.rmtree(path)


class OwnedTemporaryDirectory:
    """Private name/context/cleanup/finalizer surface needed by existing code."""
    def __init__(self,*,prefix='tmp',dir=None):
        path=directory(prefix=prefix,dir=dir)
        self.name=str(path);self.parent=str(path.parent)
        current=path.lstat();self.identity=(current.st_dev,current.st_ino)
        self._cleanup_attempted=False
        self._finalizer=weakref.finalize(self,_remove_owned,self.name,self.parent,self.identity)

    def __enter__(self):return self.name

    def cleanup(self):
        if self._cleanup_attempted:return
        self._cleanup_attempted=True
        self._finalizer.detach()
        _remove_owned(self.name,self.parent,self.identity)

    def __exit__(self,kind,error,traceback):
        if error is None:
            self.cleanup()
        else:
            try:self.cleanup()
            except BaseException as cleanup_error:
                BaseException.add_note(error,'Owned temporary cleanup also failed: '+type(cleanup_error).__name__)
        return False
