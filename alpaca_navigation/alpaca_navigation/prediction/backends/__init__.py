"""Prediction backends vendored into this package.

Each entry maps a backend name to the module and class implementing it, so
adding a backend is a data change. Imports stay lazy: the heavy model code is
only loaded for the backend actually requested.
"""

from importlib import import_module

BACKENDS = {
    'eqmotion': ('alpaca_navigation.prediction.backends.eqmotion_inference', 'EqMotionPredictor'),
    'autobots': ('alpaca_navigation.prediction.backends.autobots_inference', 'AutoBotsPredictor'),
    'moflow': ('alpaca_navigation.prediction.backends.moflow_inference', 'MoFlowPredictor'),
}


def load_predictor_class(name: str):
    """Return the predictor class for `name`, importing it on first use."""
    try:
        module_path, class_name = BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"Unknown prediction backend {name!r}. Available: {sorted(BACKENDS)}"
        ) from None

    try:
        module = import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Prediction backend {name!r} needs the python package {exc.name!r}, "
            "which is not installed in this environment."
        ) from exc

    return getattr(module, class_name)
