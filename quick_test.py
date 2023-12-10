import logging

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.utils.estimator_checks import check_estimator

from tabpfn_client import tabpfn_classifier, UserDataClient
from tabpfn_client.tabpfn_classifier import TabPFNClassifier


if __name__ == "__main__":
    # set logging level to debug
    # logging.basicConfig(level=logging.DEBUG)

    use_server = True
    # use_server = False

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    if not use_server:
        tabpfn_classifier.init(use_server=False)
        tabpfn = TabPFNClassifier(device="cpu", N_ensemble_configurations=4)
        # check_estimator(tabpfn)
        tabpfn.fit(X_train, y_train)
        print("predicting")
        print(tabpfn.predict(X_test))
        print("predicting_proba")
        print(tabpfn.predict_proba(X_test))

    else:
        tabpfn_classifier.init()
        tabpfn = TabPFNClassifier(device="cpu", N_ensemble_configurations=4)
        # print("checking estimator", check_estimator(tabpfn))
        tabpfn.fit(X_train, y_train)
        print("predicting")
        print(tabpfn.predict(X_test))
        print("predicting_proba")
        print(tabpfn.predict_proba(X_test))

        print(UserDataClient().get_data_summary())