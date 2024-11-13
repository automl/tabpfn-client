# The client for the (all new) TabPFN

This is an alpha family and friends service, so please do not expect this to never be down or run into errors. It worked fine in the settings that we tried, though.

What model is behind the API? It is a new TabPFN which we allow to handle up to 10K data points with up to 500 features. You can control all pre-processing, the amount of ensembling etc.

### We would really appreciate your feedback! Please join our discord community here https://discord.gg/VJRuU3bSxt or email us at hello@priorlabs.ai


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

- To create a new account, simply call the init() or complete the registration process during tabpfn.fit(). 

- Once registered, you will get the information about where your access token is stored, which you can also get by get_token() if it exists.

- Have a valid access token? Directly login using:
init(access_token="valid_access_token")

```

Now you can use our model just like any other sklearn estimator
```python
tabpfn = TabPFNClassifier()
tabpfn.fit(X_train, y_train)
tabpfn.predict(X_test)
# or you can also use tabpfn.predict_proba(X_test)
```

To login using your access token, skipping the interactive flow, use:

```python
from tabpfn_client import config

# Retrieve Token
get_token()
```

```python
from tabpfn_client import config

# Set Token
init(access_token="valid_access_token")
```

# Development

To encourage better coding practices, `ruff` has been added to the pre-commit hooks. This will ensure that the code is formatted properly before being committed. To enable pre-commit (if you haven't), run the following command:
```sh
pre-commit install
```
Additionally, it is recommended that developers install the ruff extension in their preferred editor. For installation instructions, refer to the [Ruff Integrations Documentation](https://docs.astral.sh/ruff/integrations/).
