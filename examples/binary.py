"""User-supplied data only;running this example performs real training."""
from morphtree import MorphTree
def train(X_train,y_train,X_test):
    model=MorphTree(task='binary',random_state=42,verbosity=0)
    model.fit(X_train,y_train)
    return model,model.predict(X_test)
