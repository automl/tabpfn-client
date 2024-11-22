# The client for the (all new) TabPFN

This is an alpha family and friends service, so please do not expect this to never be down or run into errors. It worked fine in the settings that we tried, though.

What model is behind the API? It is a new TabPFN which we allow to handle up to **10K data points** with up to **500 features** for both **regression** and **classification**. You can control all pre-processing, the amount of ensembling etc.

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
init()
```

Now you can use our model just like any other sklearn estimator
```python
tabpfn = TabPFNClassifier()
tabpfn.fit(X_train, y_train)
tabpfn.predict(X_test)
# or you can also use tabpfn.predict_proba(X_test)
```

You can now retrieve your access token

```python
import tabpfn_client
token = tabpfn_client.get_access_token()
```

and login (on another machine) using your access token, skipping the interactive flow, use:

```python
tabpfn_client.set_access_token(token)
```

# Development

To encourage better coding practices, `ruff` has been added to the pre-commit hooks. This will ensure that the code is formatted properly before being committed. To enable pre-commit (if you haven't), run the following command:
```sh
pre-commit install
```
Additionally, it is recommended that developers install the ruff extension in their preferred editor. For installation instructions, refer to the [Ruff Integrations Documentation](https://docs.astral.sh/ruff/integrations/).


### Build for PyPI

```bash
if [ -d "dist" ]; then rm -rf dist/*; fi
python3 -m pip install --upgrade build; python3 -m build
python3 -m twine upload --repository pypi dist/*
```


### Access/Delete Personal Information

You can use our `UserDataClient` to access and delete personal information.

```python
from tabpfn_client import UserDataClient

print(UserDataClient.get_data_summary())
```
