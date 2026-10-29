"""Explicit software driver, never an Optuna study/trial or a fit."""
from types import SimpleNamespace


def module(fault=None):
    class Trial:
        def __init__(self,number):self.number=number
        def suggest_int(self,name,low,high):return low+1+self.number
        def suggest_float(self,name,low,high,log=False):return low if self.number==0 else high
    class Study:
        def __init__(self):self.n=0
        def ask(self):
            if fault=='ask':raise RuntimeError('SIMULATED_ASK_FAILURE')
            trial=Trial(self.n);self.n+=1;return trial
        def tell(self,*a,**k):
            if fault=='tell':raise RuntimeError('SIMULATED_TELL_FAILURE')
    class Sampler:
        def __init__(self,**kwargs):self.parameters=kwargs
    return SimpleNamespace(samplers=SimpleNamespace(TPESampler=Sampler),
        trial=SimpleNamespace(TrialState=SimpleNamespace(FAIL='SIMULATED_FAIL')),
        create_study=lambda **kwargs:Study())
