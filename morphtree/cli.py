"""Inspect, predict and export saved public MorphTree models (no training)."""
import argparse,json
from pathlib import Path
from .api import MorphTree
from ._experimental.target import _NamedRows
from ._experimental.roundtrip_v3 import decode
from ._experimental.admission import sha

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('inspect','predict','export'))
    p.add_argument('--model',required=True);p.add_argument('--sha256')
    p.add_argument('--input');p.add_argument('--output')
    a=p.parse_args(argv)
    try:
        m=MorphTree.load(a.model,sha256=a.sha256)
        if a.command=='inspect':v={'program':m.get_algorithm_program(),'recipe':m.get_recipe(),'history':m.get_mutation_history()}
        elif a.command=='export':
            if not a.output:raise ValueError('--output required')
            m.export_standalone(a.output);v={'standalone':a.output,'new_fits':0}
        else:
            if not a.input:raise ValueError('--input required')
            f=Path(a.input)
            if not 0<f.stat().st_size<=2*1024**2:raise ValueError('Input JSON must be within2MiB')
            v=decode(f.read_bytes(),sha(f))
            x=_NamedRows(v['rows'],v['columns']) if type(v) is dict and set(v)=={'columns','rows'} else v
            v={'prediction':m.predict(x)}
            if m.task_!='regression':v['probabilities']=m.predict_proba(x)
        print(json.dumps(v,ensure_ascii=False,allow_nan=False));return 0
    except (ValueError,TypeError,OSError,RuntimeError,ImportError) as e:p.exit(2,type(e).__name__+': '+str(e)+'\n')
if __name__=='__main__':raise SystemExit(main())
