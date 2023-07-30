import pathlib

from omegaconf import OmegaConf

from tabpfn import TabPFNClassifier as TabPFNClassifierLocal
from tabpfn_client.tabpfn_classifier_interface import TabPFNClassifierInterface
from tabpfn_client.tabpfn_service_client import TabPFNServiceClient

SERVER_SPEC_FILE = pathlib.Path(__file__).parent.resolve() / "server_spec.yaml"

g_use_hosted_tabpfn = True
g_is_initialized = False
g_access_token = None


def init_tabpfn_builder(access_token=None, local=False):
    if not local:
        if access_token is None:
            # look for access_token that is stored in our package
            # if it is not there, throw error
            raise ValueError("access_token must be provided to use our hosted TabPFNClassifier service")

        # temp
        global g_access_token
        g_access_token = access_token

        # authenticate access_token against our hosted TabPFNClassifier service
        # if it is not valid, throw error

        # check if this access_token is already saved in our package,
        # if not, save it to disk in our package using __file__

        # warn user that this access_token is saved in our package,
        # and it is assumed that there is only one user per python environment

    else:
        # set global use_hosted_tabpfn to False
        global g_use_hosted_tabpfn
        g_use_hosted_tabpfn = False

    global g_is_initialized
    g_is_initialized = True


def remove_saved_access_token():
    pass


class TabPFNClassifier(TabPFNClassifierInterface):
    # TODO: ask Sam/Noah if we could create an interface of TabPFNClassifier instead

    def __init__(self, device='cpu', base_path=pathlib.Path(__file__).parent.parent.resolve(), model_string='',
                 N_ensemble_configurations=3, no_preprocess_mode=False, multiclass_decoder='permutation',
                 feature_shift_decoder=True, only_inference=True, seed=0, no_grad=True, batch_size_inference=32):

        if not g_is_initialized:
            raise RuntimeError("init_tabpfn_builder() must be called before using TabPFNClassifier")

        if g_use_hosted_tabpfn:
            server_specs = OmegaConf.load(SERVER_SPEC_FILE)
            self.classifier = TabPFNServiceClient(server_specs, g_access_token)

        else:
            self.classifier = TabPFNClassifierLocal(device, base_path, model_string, N_ensemble_configurations,
                                                    no_preprocess_mode, multiclass_decoder, feature_shift_decoder,
                                                    only_inference, seed, no_grad, batch_size_inference)

    def remove_models_from_memory(self):
        return self.classifier.remove_models_from_memory()

    def load_result_minimal(self, path, i, e):
        return self.classifier.load_result_minimal(path, i, e)

    def fit(self, X, y):
        return self.classifier.fit(X, y)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def predict(self, X, return_winning_probability=False, normalize_with_test=False):
        return self.classifier.predict(X, return_winning_probability, normalize_with_test)

    def try_root(self):
        return self.classifier.try_root()
