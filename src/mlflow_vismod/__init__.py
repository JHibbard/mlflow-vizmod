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
import shutil
import importlib
import pkgutil

# External Libraries
import yaml
import mlflow
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow import pyfunc
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.file_utils import _copy_file_or_tree
from mlflow.utils.annotations import keyword_only
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import DIRECTORY_NOT_EMPTY
from mlflow.tracking.artifact_utils import _download_artifact_from_uri


# Internal Libraries
import mlflow_vismod
import mlflow_vismod.styles


FLAVOR_NAME = 'mlflow_vismod'
MODEL_DIR_SUBPATH = 'viz'

_SERIALIZED_VEGA_MODEL_FILE_NAME = ''
_PICKLE_MODULE_INFO_FILE_NAME = 'pickle_module_info.txt'

_logger = logging.getLogger(__name__)


def iter_namespace(ns_pkg):
    # Specifying the second argument (prefix) to iter_modules makes the
    # returned name an absolute name instead of a relative one. This allows
    # import_module to work without having to do additional modification to
    # the name.
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")


_discovered_styles = {
    name: importlib.import_module(name)
    for finder, name, ispkg
    in iter_namespace(mlflow_vismod.styles)
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
        additional_pip_deps=['altair==4.1.0', f'mlflow.styles.{style}'],
        additional_conda_channels=None,
    )


@keyword_only
def log_model(
    model,  # vega_saved_model_dir,
    artifact_path,
    conda_env=None,
    signature=None,
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
        signature=signature,
        input_example=input_example,
        style=style,
    )


@keyword_only
def save_model(
    model,  # vega_saved_model_dir=vega_saved_model_dir,
    path,
    mlflow_model=None,
    conda_env=None,
    signature=None,
    input_example=None,
    style=None,
):
    """
    """
    if os.path.exists(path):
        raise MlflowException("Path '{}' already exists".format(path), DIRECTORY_NOT_EMPTY)
    os.makedirs(path)
    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        # tensorflow -> _save_example(mlflow_model, input_example, path)
        # sklearn
        mlflow_model.signature = signature
    # root_relative_path = _copy_file_or_tree(src=artifact_path, dst=path, dst_dir=None)
    # shutil.move(os.path.join(path, root_relative_path), os.path.join(path, MODEL_DIR_SUBPATH))

    # Style-specific Saving
    current_style = _discovered_styles[f'mlflow_vismod.styles.{style}']
    mlflow.log_param('path', path)
    current_style.Style.save(model, 'vizzy.html')
    root_relative_path = _copy_file_or_tree(src='vizzy.html', dst=os.path.join(path, 'vizzy.html'), dst_dir=None)
    shutil.move(os.path.join(path, root_relative_path), os.path.join(path, MODEL_DIR_SUBPATH))

    conda_env_subpath = 'conda.yaml'
    if conda_env is None:
        conda_env = get_default_conda_env(style=style)
    elif not isinstance(conda_env, dict):
        with open(conda_env, 'r') as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        saved_model_dir=MODEL_DIR_SUBPATH,
    )
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow_vismod", env=conda_env_subpath)
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))


def load_model(model_uri, style=None):
    """
    """
    current_style = _discovered_styles[f'mlflow_vismod.styles.{style}']
    return current_style.Style(
        artifact_uri=_download_artifact_from_uri(artifact_uri=model_uri),
    )
