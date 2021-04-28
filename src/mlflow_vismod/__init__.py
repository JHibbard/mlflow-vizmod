"""
The `mlflow_vismod` module provides an API for logging and loading Vega models. This module
exports Vega models with the following flavors:

Vega (native) format
    This is the main flavor that can be loaded back into Vega.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
"""
# Standard Libraries
import logging
import os
import importlib
import pkgutil

# External Libraries
import yaml
import mlflow
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow import pyfunc
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.annotations import keyword_only
from mlflow.models.utils import _save_example
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import DIRECTORY_NOT_EMPTY
from mlflow.tracking.artifact_utils import _download_artifact_from_uri


# Internal Libraries
import mlflow_vismod
import mlflow_vismod.styles


FLAVOR_NAME = "mlflow_vismod"
MODEL_DIR_SUBPATH = "viz"
SERIALIZATION_FORMAT_PICKLE = "pickle"
SERIALIZATION_FORMAT_CLOUDPICKLE = "cloudpickle"
SUPPORTED_SERIALIZATION_FORMATS = [
    SERIALIZATION_FORMAT_PICKLE,
    SERIALIZATION_FORMAT_CLOUDPICKLE,
]

_logger = logging.getLogger(__name__)


def iter_namespace(ns_pkg):
    # Specifying the second argument (prefix) to iter_modules makes the
    # returned name an absolute name instead of a relative one. This allows
    # import_module to work without having to do additional modification to
    # the name.
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")


_discovered_styles = {
    name: importlib.import_module(name)
    for finder, name, ispkg in iter_namespace(mlflow_vismod.styles)
}


def discovered_styles():
    return _discovered_styles


def get_default_conda_env(style):
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """

    return _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=["altair==4.1.0", f"mlflow.styles.{style}"],
        additional_conda_channels=None,
    )


@keyword_only
def log_model(
    model,  # vega_saved_model_dir,
    artifact_path,
    conda_env=None,
    # signature=None,
    input_example=None,
    registered_model_name=None,
    style=None,
):
    """

    :return:
    """
    return Model.log(
        model=model,  # vega_saved_model_dir=vega_saved_model_dir,
        artifact_path=artifact_path,
        flavor=mlflow_vismod,
        conda_env=conda_env,
        registered_model_name=registered_model_name,
        # signature=signature,
        input_example=input_example,
        style=style,
    )


@keyword_only
def save_model(
    model,  # vega_saved_model_dir=vega_saved_model_dir,
    path,
    mlflow_model=None,
    conda_env=None,
    # signature=None,
    input_example=None,
    style=None,
):
    """Save a visual model to a local file or a run.


    """
    if os.path.exists(path):
        raise MlflowException(
            "Path '{}' already exists".format(path), DIRECTORY_NOT_EMPTY
        )
    os.makedirs(path)
    if mlflow_model is None:
        mlflow_model = Model()
    # if signature is not None:
    #     mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)

    # Style-specific Save Logic
    current_style = _discovered_styles[f"mlflow_vismod.styles.{style}"]
    current_style.Style.save(model, path)

    # Saving Conda Environment
    conda_env_subpath = "conda.yaml"
    if conda_env is None:
        conda_env = get_default_conda_env(style=style)
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        # saved_model_dir=MODEL_DIR_SUBPATH,
        pickled_model="viz.pkl",
    )
    pyfunc.add_to_model(
        mlflow_model, loader_module="mlflow_vismod", env=conda_env_subpath
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))


def load_model(model_uri, style=None):
    """Load a visual model from a local file or a run.

    :param model_uri: The location, in URI format, of the MLflow model, for example:
                  - ``/Users/me/path/to/local/model``
                  - ``relative/path/to/local/model``
                  - ``s3://my_bucket/path/to/model``
                  - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                  - ``models:/<model_name>/<model_version>``
                  - ``models:/<model_name>/<stage>``
                  For more information about supported URI schemes, see
                  `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
                  artifact-locations>`_.
    :return: A visual model.
    """
    current_style = _discovered_styles[f"mlflow_vismod.styles.{style}"]
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    flavor_conf = _get_flavor_configuration(
        model_path=local_model_path, flavor_name=FLAVOR_NAME
    )
    vismod_model_artifacts_path = os.path.join(
        local_model_path, flavor_conf["pickled_model"]
    )
    serialization_format = flavor_conf.get(
        "serialization_format", SERIALIZATION_FORMAT_CLOUDPICKLE
    )
    return current_style.Style(
        artifact_uri=vismod_model_artifacts_path,
    )
