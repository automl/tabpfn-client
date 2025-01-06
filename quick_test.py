import logging

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split

from tabpfn_client import UserDataClient
from tabpfn_client.estimator import TabPFNClassifier, TabPFNRegressor

logging.basicConfig(level=logging.DEBUG)


if __name__ == "__main__":
    # set logging level to debug
    # logging.basicConfig(level=logging.DEBUG)

    use_server = True
    # use_server = False

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    # make y text
    y_train = y_train.astype(str) + "a"
    y_test = y_test.astype(str) + "a"

    tabpfn = TabPFNClassifier(n_estimators=3)
    # print("checking estimator", check_estimator(tabpfn))
    tabpfn.fit(X_train[:99], y_train[:99])
    print("predicting")
    print(tabpfn.predict(X_test))
    print("predicting_proba")
    print(tabpfn.predict_proba(X_test))

    print(UserDataClient.get_data_summary())

    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    tabpfn = TabPFNRegressor(n_estimators=3)
    # print("checking estimator", check_estimator(tabpfn))
    tabpfn.fit(X_train[:99], y_train[:99])
    print("predicting reg")
    print(tabpfn.predict(X_test, output_type="mean"))

    print(UserDataClient.get_data_summary())
    # test predict_full
    print(tabpfn.predict(X_test[:30], output_type="full", quantiles=[0.1, 0.5, 0.9]))
