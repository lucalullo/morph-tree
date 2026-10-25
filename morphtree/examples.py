"""Normal user code, executed by an approved host in this qualification build.

No Session, checkpoint, campaign identifier or research directory is an API
argument. Current standalone training remains disabled until a run is approved.
The same function works for binary, multiclass and regression targets.
"""
from morphtree import MorphTree


def train(X_train, y_train, *, task='auto', time_budget=300):
    model=MorphTree(task=task,mode='baseline',random_state=42,
                    tuning='none',time_budget=time_budget)
    model.fit(X_train,y_train)
    return model


def save_and_predict(model, X_test, path, export_path):
    model.save(path)
    restored=MorphTree.load(path)
    predictions=restored.predict(X_test)
    probabilities=(restored.predict_proba(X_test)
                   if restored.task_!='regression' else None)
    restored.export_python(export_path)
    return {'predictions':predictions,'probabilities':probabilities,
            'recipe':restored.recipe(),'configuration':restored.get_params()}
