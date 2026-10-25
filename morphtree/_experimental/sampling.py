import random
def bootstrap(y, task, seed):
    rng = random.Random(seed)
    if task == 'regression':
        return [rng.randrange(len(y)) for _ in y]
    groups = [[i for i, v in enumerate(y) if v == c] for c in sorted(set(y))]
    ids = [rng.choice(group) for group in groups for _ in group]
    rng.shuffle(ids)
    return ids

def oof_folds(y, task, seed):
    groups = [list(range(len(y)))] if task == 'regression' else [[i for i, v in enumerate(y) if v == c] for c in sorted(set(y))]
    rng = random.Random(seed)
    folds = [[], []]
    for group in groups:
        if len(group) < 2:
            raise ValueError('OOF requires at least two examples in every TRAIN class')
        rng.shuffle(group)
        for j, i in enumerate(group):
            folds[j % 2].append(i)
    return [sorted(v) for v in folds]
