from typing import Optional, Tuple, Literal, Dict
import logging
from dataclasses import dataclass, asdict

from tabpfn_client import config
from tabpfn_client.estimator import TabPFNClassifier, TabPFNRegressor

logger = logging.getLogger(__name__)

class CAAFE():
    def __init__(
        self,
        model: str,
        n_iters: int,
        caafe_method: str = "base",

    ):
        self.model = model