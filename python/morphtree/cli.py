"""Inspect, predict and export saved public MorphTree models (no training)."""
import argparse,json
from pathlib import Path
from . import load_model
from .persistence import regular_file_size,read_regular_bytes,decode_json_bytes,LoadLimits


class _NamedRows:
    """JSON rows with column names for both supported model generations."""
    def __init__(self,rows,names):
        self.rows=rows;self.columns=tuple(names);self.shape=(len(rows),len(names))
    def __len__(self):return len(self.rows)
    def to_numpy(self):return self.rows
    def itertuples(self,index=False,name=None):return iter(self.rows)
    @property
    def iloc(self):return self
    def __getitem__(self,key):
        class Column:
            def __init__(self,values):self.array=values
        return Column([row[key[1]] for row in self.rows[key[0]]])

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('inspect','predict','export'))
    p.add_argument('--model',required=True);p.add_argument('--sha256')
    p.add_argument('--input');p.add_argument('--output')
    a=p.parse_args(argv)
    try:
        m=load_model(a.model,sha256=a.sha256)
        if a.command=='inspect':v={'program':m.get_algorithm_program(),'recipe':m.get_recipe(),'history':m.get_mutation_history()}
        elif a.command=='export':
            if not a.output:raise ValueError('--output required')
            m.export_standalone(a.output);v={'standalone':a.output,'new_fits':0}
        else:
            if not a.input:raise ValueError('--input required')
            f=Path(a.input)
            from .scalable import managed_resources
            from .resource_plan import io_seconds
            size=regular_file_size(f);res=managed_resources('auto',io_seconds(size))
            if size<1:raise ValueError('Nonempty regular JSON input required')
            res.reserve(size*12+4096,'CLI JSON input workspace')
            raw=read_regular_bytes(f,max(1,res.bytes_limit//12))
            positional=raw.lstrip().startswith(b'[')
            if positional:raw=b'{"rows":'+raw+b'}'
            v=decode_json_bytes(raw,LoadLimits(max_bytes=max(1,res.bytes_limit//12)+9),reserve=res.reserve)
            if positional:v=v['rows']
            x=_NamedRows(v['rows'],v['columns']) if type(v) is dict and set(v)=={'columns','rows'} else v
            v={'prediction':m.predict(x)}
            if m.task_!='regression':v['probabilities']=m.predict_proba(x)
        print(json.dumps(v,ensure_ascii=False,allow_nan=False));return 0
    except (ValueError,TypeError,OSError,RuntimeError,ImportError,MemoryError) as e:p.exit(2,type(e).__name__+': '+str(e)+'\n')
if __name__=='__main__':raise SystemExit(main())
