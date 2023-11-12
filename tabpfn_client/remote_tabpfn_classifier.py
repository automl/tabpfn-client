from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin

from tabpfn_client.service_wrapper import InferenceClient


class RemoteTabPFNClassifier(BaseEstimator, ClassifierMixin):

    def __init__(
            self,
            model=None,
            device="cpu",
            base_path=".",
            model_string="",
            batch_size_inference=4,
            fp16_inference=False,
            inference_mode=True,
            c=None,
            N_ensemble_configurations=10,
            preprocess_transforms=("none", "power_all"),
            feature_shift_decoder=False,
            normalize_with_test=False,
            average_logits=False,
            categorical_features=tuple(),
            optimize_metric=None,
            seed=None,
            transformer_predict_kwargs_init=None,
            multiclass_decoder="permutation",

            # dependency injection (for testing)
            inference_handler=InferenceClient()
    ):
        # TODO:
        #  These configs are ignored at the moment -> all clients share the same (default) on-server TabPFNClassifier.
        #  In the future version, these configs will be used to create per-user TabPFNClassifier,
        #    allowing the user to setup the desired TabPFNClassifier on the server.
        # config for tabpfn
        self.model = model
        self.device = device
        self.base_path = base_path
        self.model_string = model_string
        self.batch_size_inference = batch_size_inference
        self.fp16_inference = fp16_inference
        self.inference_mode = inference_mode
        self.c = c
        self.N_ensemble_configurations = N_ensemble_configurations
        self.preprocess_transforms = preprocess_transforms
        self.feature_shift_decoder = feature_shift_decoder
        self.normalize_with_test = normalize_with_test
        self.average_logits = average_logits
        self.categorical_features = categorical_features
        self.optimize_metric = optimize_metric
        self.seed = seed
        self.transformer_predict_kwargs_init = transformer_predict_kwargs_init
        self.multiclass_decoder = multiclass_decoder

        self.inference_handler = inference_handler

    def fit(self, X, y):
        self.inference_handler.fit(X, y)
        self.fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.inference_handler.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self)
        return self.inference_handler.predict_proba(X)
