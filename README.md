# PLEASE READ: Alpha Client for the Updated TabPFN

Password for registration / validation link: tabpfn-2023

This is an alpha family and friends service, so please do not expect this to never be down or run into errors.
We did test it though and can say that it seems to work fine in the settings that we tried.

PLEASE DO NOT SHARE THIS REPOSITORY at this point outside of the NeurIPS Tabular Representation workshop.

What model is behind the API? For now, this version is the light version to save compute on our side, not the TabPFN (Fast) or TabPFN (Best-Q) which we presented at the Neurips Workshop. We will change this once we see our server is working stably. It is a new TabPFN which we allow to handle up to 10K instances with up to 500 features.
This TabPFN is not ensembled, we will put out improved and ensembled models soon.

### We would really appreciate your feedback! If you encounter bugs or suggestions for improvement please create an issue or email me (samuelgabrielmuller (at) gmail com).


# How To

### Tutorial

[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ns_KdtyHgl29AOVwTw9c-DZrPj7fx_DW?usp=sharing)

We created a [colab](https://colab.research.google.com/drive/1ns_KdtyHgl29AOVwTw9c-DZrPj7fx_DW?usp=sharing)
tutorial to get started quickly.

### Installation

```bash
pip install tabpfn-client
```

### Usage

Import and login
```python
from tabpfn_client import init, TabPFNClassifier
init()
```

Now you can use our model just like any other sklearn estimator
```python
tabpfn = TabPFNClassifier()
tabpfn.fit(X_train, y_train)
tabpfn.predict(X_test)
# or you can also use tabpfn.predict_proba(X_test)
```
