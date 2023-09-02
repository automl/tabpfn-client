from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from tabpfn_client import tabpfn_classifier
from tabpfn_client.tabpfn_classifier import TabPFNClassifier


if __name__ == "__main__":
    use_server = True
    # use_server = False

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    if not use_server:
        tabpfn_classifier.init(use_server=False)
        tabpfn = TabPFNClassifier(device="cpu", N_ensemble_configurations=4)
        tabpfn.fit(X_train, y_train)
        tabpfn.predict(X_test)
        tabpfn.predict_proba(X_test)

    else:
        tabpfn_classifier.init()
        tabpfn = tabpfn_classifier.TabPFNClassifier()
        tabpfn.fit(X_train, y_train)
        tabpfn.predict(X_test)

        # not supported yet
        # tabpfn.predict_proba(X_test)
