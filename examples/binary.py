"""User-supplied data only;running this example performs real training."""
from morphtree import MorphTreeClassifier
def train(X_train,y_train,X_test):
    model=MorphTreeClassifier(random_state=42)
    model.fit(X_train,y_train)
    return model,model.predict(X_test)
