from typing import Optional, Dict, Literal
import logging

from tabpfn_client import config

logger = logging.getLogger(__name__)


class CAAFE:
    def __init__(
        self,
        model: str,
        n_iters: int,
        caafe_method: str = "base",
        params: Optional[Dict] = None,
    ):
        self.model = model
        self.n_iters = n_iters
        self.caafe_method = caafe_method
        self.data_description = None
        self.params = params

        config.init()

    def fit(self, X, y, data_description):
        assert data_description is not None, "Data description needs to be provided"
        self.data_description = data_description
        # assert init() is called
        if not config.g_caafe_config.is_initialized:
            raise RuntimeError("tabpfn_client.init() must be called before using CAAFE")

        if config.g_caafe_config.use_server:
            try:
                assert (
                    self.model == "latest_tabpfn_hosted"
                ), "Only 'latest_tabpfn_hosted' model is supported at the moment for init(use_server=True)"
            except AssertionError as e:
                print(e)
            uid = config.g_caafe_config.inference_handler.fit(
                X, y, self.data_description
            )
            print(f"uid: {uid}")
            self.fitted_ = True
        else:
            raise NotImplementedError(
                "Only server mode is supported at the moment for init(use_server=False)"
            )
        return self

    def generate_features(
        self, task: Literal["classification", "regression"] = "classification"
    ):
        if not self.fitted_:
            raise RuntimeError("Model has not been fitted yet")
        return config.g_caafe_config.inference_handler.generate_features(
            config=self.params,
            task=task,
        )
