import logging

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split

from tabpfn_client import UserDataClient
from tabpfn_client.caafe_generator import CAAFE

logging.basicConfig(level=logging.DEBUG)

caafe_config = {
    "model": "gpt-3.5-turbo",
    "iterative": 3,
    "metric_used": "auc",
    "display_method": "print",
    "n_splits": 3,
    "n_repeats": 1,
    "markdown_prompt": False,
    "test_type": "ttest",
    "check_significance": True,
    "check_corr": False,
}

if __name__ == "__main__":
    # set logging level to debug
    # logging.basicConfig(level=logging.DEBUG)

    use_server = True
    # use_server = False

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    dataset_desc = """
    Diabetes patient records were obtained from two sources:  an automatic electronic recording device and paper records.
    The automatic device had an internal clock to timestamp events, whereas the paper records only provided "logical time" slots (breakfast, lunch, dinner, bedtime).
    For paper records, fixed times were assigned to breakfast (08:00), lunch (12:00), dinner (18:00), and bedtime (22:00).
    Thus paper records have fictitious uniform recording times whereas electronic records have more realistic time stamps.
    """

    caafe = CAAFE(model="latest_tabpfn_hosted", n_iters=3, params=caafe_config)
    caafe.fit(X_train[:100], y_train[:100], dataset_desc)
    print("generating using caafe")
    res = caafe.generate_features(task="regression")

    print(res)
    print(UserDataClient().get_data_summary())

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    dataset_desc = """
    Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.
n the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].

This database is also available through the UW CS ftp server:
ftp ftp.cs.wisc.edu
cd math-prog/cpo-dataset/machine-learn/WDBC/

Also can be found on UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
    """

    caafe = CAAFE(model="latest_tabpfn_hosted", n_iters=3, params=caafe_config)
    # print("checking estimator", check_estimator(tabpfn))
    caafe.fit(X_train[:100], y_train[:100], dataset_desc)
    print("generating using caafe")
    res = caafe.generate_features(task="classification")

    print(res)
    print(UserDataClient().get_data_summary())
