"""Inference-only CLI for the integrated experimental build."""
import argparse
import json
from pathlib import Path
from .product import MorphTree
from .target import _NamedRows
from .roundtrip_v3 import decode
import hashlib


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('inspect', 'predict', 'export'))
    parser.add_argument('--model', required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--workdir', required=True)
    parser.add_argument('--input')
    parser.add_argument('--output')
    args = parser.parse_args(argv)
    try:
        model = MorphTree.load(args.model, sha256=args.sha256, workdir=args.workdir)
        if args.command == 'inspect':
            value = {'task': model.task_, 'family': model.best_family_,
                     'recipe': model.recipe(), 'cost': model.cost_report()}
        elif args.command == 'predict':
            if not args.input: raise ValueError('--input JSON rows required')
            path = Path(args.input)
            if not 0 < path.stat().st_size <= 16*1024**2: raise ValueError('Bounded rows required')
            raw = path.read_bytes(); rows = decode(raw, hashlib.sha256(raw).hexdigest())
            if type(rows) is dict and set(rows) == {'columns', 'rows'}:
                rows = _NamedRows(rows['rows'], rows['columns'])
            value = {'predictions': model.predict(rows)}
            if model.task_ != 'regression': value['probabilities'] = model.predict_proba(rows)
        else:
            if not args.output: raise ValueError('--output required')
            model.export_python(args.output); value = {'standalone': args.output, 'new_fits': 0}
        print(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, type(exc).__name__+': '+str(exc)+'\n')


if __name__ == '__main__':
    main()
