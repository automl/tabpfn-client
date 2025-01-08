# TabPFN Client
[![PyPI version](https://badge.fury.io/py/tabpfn-client.svg)](https://badge.fury.io/py/tabpfn-client)
[![Discord](https://img.shields.io/discord/1285598202732482621?color=7289da&label=Discord&logo=discord&logoColor=ffffff)](https://discord.com/channels/1285598202732482621/)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ns_KdtyHgl29AOVwTw9c-DZrPj7fx_DW?usp=sharing)
[![Documentation](https://img.shields.io/badge/docs-priorlabs.ai-blue)](https://priorlabs.ai/)
[![Twitter Follow](https://img.shields.io/twitter/follow/Prior_Labs?style=social)](https://twitter.com/Prior_Labs)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Last Commit](https://img.shields.io/github/last-commit/automl/tabpfn-client)


TabPFN is a foundation model for tabular data that outperforms traditional methods while being dramatically faster. This client library provides easy access to the TabPFN API, enabling state-of-the-art tabular machine learning in just a few lines of code.

📚 For detailed usage examples and best practices, check out:
- [Interactive Colab Tutorial](https://colab.research.google.com/drive/1ns_KdtyHgl29AOVwTw9c-DZrPj7fx_DW?usp=sharing)

## ⚠️ Alpha Release Note
This is an alpha release. While we've tested it thoroughly in our use cases, you may encounter occasional issues. We appreciate your understanding and feedback as we continue to improve the service.

This is a cloud-based service. Your data will be sent to our servers for processing.
- Do NOT upload any Personally Identifiable Information (PII)
- Do NOT upload any sensitive or confidential data
- Do NOT upload any data you don't have permission to share
- Consider anonymizing or pseudonymizing your data before upload
- Review your organization's data sharing policies before use

## 🌐 TabPFN Ecosystem

Choose the right TabPFN implementation for your needs:

- **TabPFN Client (this repo)**: Easy-to-use API client for cloud-based inference
- **[TabPFN Extensions](https://github.com/priorlabs/tabpfn-extensions)**: Community extensions and integrations
- **[TabPFN](https://github.com/priorlabs/tabpfn)**: Core implementation for local deployment and research

## 🏁 Quick Start

### Installation
```bash
pip install tabpfn-client
```

### Basic Usage
```python
from tabpfn_client import init, TabPFNClassifier, TabPFNRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load an example dataset

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Use it like any sklearn model
model = TabPFNClassifier()
model.fit(X_train, y_train)
# Get predictions
predictions = model.predict(X_test)
# Get probability estimates
probabilities = model.predict_proba(X_test)
```

## 🔑 Authentication
### Load Your Token
```python
import tabpfn_client
token = tabpfn_client.get_access_token()
```

and login (on another machine) using your access token, skipping the interactive flow, use:
```python
tabpfn_client.set_access_token(token)
```

## 🤝 Join Our Community
We're building the future of tabular machine learning and would love your involvement! Here's how you can participate and get help:

1. **Try TabPFN**: Use it in your projects and share your experience
2. **Connect & Learn**: 
   - Join our [Discord Community](https://discord.gg/VJRuU3bSxt) for discussions and support
   - Read our [Documentation](https://priorlabs.ai/) for detailed guides
   - Check out [GitHub Issues](https://github.com/automl/tabpfn-client/issues) for known issues and feature requests
3. **Contribute**: 
   - Report bugs or request features through issues
   - Submit pull requests (see development guide below)
   - Share your success stories and use cases
4. **Stay Updated**: Star the repo and join Discord for the latest updates


## 📊 Usage Limits

### API Cost Calculation
Each API request consumes usage credits based on the following formula:
```python
api_cost = (num_train_rows + num_test_rows) * num_cols * n_estimators
```
Where `n_estimators` defaults to:
- 4 for classification tasks
- 8 for regression tasks

Per day the current prediction allowance is 5,000,000 cells. We will adjust this limit based on usage patterns.

### Monitoring Usage
Track your API usage through response headers:
- `X-RateLimit-Limit`: Your total allowed usage
- `X-RateLimit-Remaining`: Remaining usage
- `X-RateLimit-Reset`: Reset timestamp (UTC)

Usage limits reset daily at 00:00:00 UTC.

### Size Limitations

1. Maximum total cells per request must be below 100,000:
```python
max_cells = (num_train_rows + num_test_rows) * num_cols
```

2. For regression with full output (`return_full_output=True`), the number of test samples must be below 500:
```python
if task == 'regression' and return_full_output and num_test_samples > 500:
    raise ValueError("Cannot return full output for regression with >500 test samples")
```

These limits will be increased in future releases.

## Access/Delete Personal Information

You can use our `UserDataClient` to access and delete personal information.

```python
from tabpfn_client import UserDataClient

print(UserDataClient.get_data_summary())
```


## 🤝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE.txt](LICENSE.txt) file for details.


## Development
To encourage better coding practices, `ruff` has been added to the pre-commit hooks. This will ensure that the code is formatted properly before being committed. To enable pre-commit (if you haven't), run the following command:

```bash
pre-commit install
```


Additionally, it is recommended that developers install the ruff extension in their preferred editor. For installation instructions, refer to the [Ruff Integrations Documentation](https://docs.astral.sh/ruff/integrations/).

### Build from GitHub
```bash
!git clone https://github.com/automl/tabpfn-client
%cd tabpfn-client
!git submodule update --init --recursive
!pip install -e .
%cd ..
```

### Build for PyPI
```bash
if [ -d "dist" ]; then rm -rf dist/*; fi
python3 -m pip install --upgrade build; python3 -m build
python3 -m twine upload --repository pypi dist/*
```
