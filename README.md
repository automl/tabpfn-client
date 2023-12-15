# tabpfn-client

## Preview / alpha release
Thank you for looking into our project! PLEASE DO NOT SHARE THIS REPOSITORY at this point outside of the NeurIPS Tabular Representation workshop, as we are still working on it. The API provided here and our prediction server are in a preview stage. We hope everything will work smoothly and are working to improve this repository. Please let us know if you encounter any problems.

## Interactive Demo
A Colab to demonstrating the interface: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/liam-sbhoo/a78a0fab40d8940c218cf2dc3b4f2bf8/tabpfndemo.ipynb)

## Usage
```
from tabpfn_client import init, TabPFNClassifier

init()

tabpfn = TabPFNClassifier()
tabpfn.fit(X_train, y_train)

tabpfn.predict(X_test)
```
