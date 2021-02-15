# ------------ ConfigSpace mapping ---------------
import ConfigSpace as cs
from emukit.core import (
    ParameterSpace,
    ContinuousParameter,
    DiscreteParameter,
)
from math import log, exp
from typing import Any, Sequence, Union, Dict
import numpy as np


# TODO: Define more mappings
# TODO: Clean this up/update, the methodology used in HPOBenchExperimentUtils.utils.emukit_utils is cleaner

# Mappings for __parameter definitions__ in ConfigSpace to emukit's ParameterSpace
def _to_continuous(param: cs.UniformFloatHyperparameter):
    min_val, max_val = param.lower, param.upper
    if param.log:
        min_val = log(min_val)
        max_val = log(max_val)

    return ContinuousParameter(name=param.name, min_value=min_val, max_value=max_val)


def _to_discrete(param: cs.UniformIntegerHyperparameter):
    if param.log:
        min_val, max_val = log(param.lower), log(param.upper)
    else:
        min_val, max_val = param.lower, param.upper
    return ContinuousParameter(name=param.name, min_value=min_val, max_value=max_val)


_cs_to_emu_map = {
    cs.UniformFloatHyperparameter.__name__: _to_continuous,
    cs.UniformIntegerHyperparameter.__name__: _to_discrete,
}


def map_CS_to_Emu(cspace: cs.ConfigurationSpace):
    """ Given a ConfigSpace.ConfigurationSpace object, generates a compatible Emukit ParameterSpace object. """

    params = cspace.get_hyperparameters()
    new_params = [_cs_to_emu_map[type(p).__name__](p) for p in params]
    return ParameterSpace(new_params)


# Mappings for __samples__ in emukit's ParameterSpace to ConfigSpace


def _from_continuous(param: cs.UniformFloatHyperparameter, val: Any):
    return min(max(exp(val), param.lower), param.upper) if param.log else val


def _from_discrete(param: cs.UniformIntegerHyperparameter, val: Any):
    if param.log:
        return np.clip(np.rint(exp(val)).astype(int), param.lower, param.upper)
    else:
        return np.rint(val).astype(int)


_emu_to_cs_map = {
    cs.UniformFloatHyperparameter.__name__: _from_continuous,
    cs.UniformIntegerHyperparameter.__name__: _from_discrete,
}


def EmutoCSMap(cspace: cs.ConfigurationSpace):
    """ Constructs a mapping that, given a sequence of values generated from an instance of Emukit's ParameterSpace,
    returns a sequence of values compatible with the given instance of ConfigSpace.ConfigurationSpace, given that the
    ParameterSpace itself is compatible. """

    def map(values: Sequence) -> Dict:
        return {p.name: _emu_to_cs_map[type(p).__name__](p, v) for p, v in zip(cspace.get_hyperparameters(), values)}

    return map


#### Mapping a configuration to an Emukit compatible array ###


def _map_continuous_value(val, hyper: cs.UniformFloatHyperparameter):
    return log(val) if hyper.log else val


def _map_discrete_value(val, hyper: cs.UniformIntegerHyperparameter):
    if hyper.log:
        return log(val)
    else:
        return val


value_maps = {
    cs.UniformFloatHyperparameter: _map_continuous_value,
    cs.UniformIntegerHyperparameter: _map_discrete_value,
}


def configuration_CS_to_Emu(config: Union[cs.Configuration, Sequence, Dict], cspace: cs.ConfigurationSpace):
    """ Map a Configuration object config defined in the ConfigurationSpace cspace to an equivalent array of values
    compatible with the relevant Emukit parameter space. The configuration can be provided as either an object of
    type ConfigSpace.Configuration, or a sequence of normalized unit vector values, or a sequence of unnormalized
    values expected to map directly to the keys of the ConfigurationSpace. This is controlled by the vector flag. """

    hypers = cspace.get_hyperparameters()

    if not isinstance(config, cs.Configuration):
        # Construct a cs.Configuration object from the values/vector provided.
        # This automatically checks for all constraints.
        if isinstance(config, Sequence):
            # Assume vector representation was provided
            config = cs.Configuration(configuration_space=cspace, vector=config)
        elif isinstance(config, Dict):
            # Numpy can cause havoc while loading values of different datatypes since every numpy array can only have
            # one dtype for the entire array under the current data loading regime. Thus, we need to do some
            # type-correction here.
            for h, v in config.items():
                if isinstance(v, float) and isinstance(cspace.get_hyperparameter(h), cs.UniformIntegerHyperparameter):
                    config[h] = np.rint(v).astype(int)
            config = cs.Configuration(
                configuration_space=cspace,
                values=config
            )
        else:
            raise TypeError("Invalid object type %s for config. " \
                            "Check the help for configuration_CS_to_Emu() for compatible types." % str(type(config)))

    ret = [value_maps[type(h)](config.get(h.name), h) for h in hypers]
    return ret
