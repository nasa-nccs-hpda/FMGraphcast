import numpy as np
from graphcast.graphcast import ModelConfig, TaskConfig, GraphCast
from fmgraphcast.config import config_model, config_task
from fmbase.source.merra2.model import MERRA2DataInterface
from fmgraphcast.manager import run_forward, loss_fn, grads_fn
from fmbase.util.config import configure
import jax, functools, xarray as xa
configure( 'explore-test1' )

params = None
state = {}
mconfig: ModelConfig = config_model()
tconfig: TaskConfig = config_task()

datasetMgr = MERRA2DataInterface()
tsdata: xa.DataArray = datasetMgr.load_timestep( 2000, 0 )
print( f"LOADED TRAIN DATA: shape={tsdata.shape}, dims={tsdata.dims}")

tstats: xa.Dataset = datasetMgr.load_stats( varname )

# Jax doesn't seem to like passing configs as args through the jit. Passing it
# in via partial (instead of capture by closure) forces jax to invalidate the
# jit cache if you change configs.
def with_configs(fn):
  return functools.partial( fn, model_config=mconfig, task_config=tconfig)

# Always pass params and state, so the usage below are simpler
def with_params(fn):
  return functools.partial(fn, params=params, state=state)

# Our models aren't stateful, so the state is always empty, so just return the
# predictions. This is requiredy by our rollout code, and generally simpler.
def drop_state(fn):
  return lambda **kw: fn(**kw)[0]

init_jitted = jax.jit(with_configs(run_forward.init))

if params is None:
  params, state = init_jitted( rng=jax.random.PRNGKey(0), inputs=train_inputs, targets_template=train_targets, forcings=train_forcings)

loss_fn_jitted = drop_state(with_params(jax.jit(with_configs(loss_fn.apply))))
grads_fn_jitted = with_params(jax.jit(with_configs(grads_fn)))
run_forward_jitted = drop_state(with_params(jax.jit(with_configs(run_forward.apply))))

