# Free client to use an updated TabPFN

This is an alpha family and friends service, so please do not expect this to never be down or run into errors.
We did test it though and can say that it seems to work fine in the settings that we tried.

**We release this to get feedback, if you encounter bugs or curiosities please create an issue or email me (samuelgabrielmuller (at) gmail com).**


# How to

### Tutorial

We created a ![colab](https://colab.research.google.com/drive/1ns_KdtyHgl29AOVwTw9c-DZrPj7fx_DW?usp=sharing)
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
