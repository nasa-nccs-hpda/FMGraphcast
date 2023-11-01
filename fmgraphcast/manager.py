from graphcast.graphcast import ModelConfig, TaskConfig, GraphCast
from graphcast import casting, normalization
from graphcast.predictor_base import Predictor
from graphcast import autoregressive
from graphcast import casting
from graphcast import graphcast
from graphcast import normalization
from graphcast import xarray_jax
from graphcast import xarray_tree
import haiku as hk
import jax
import functools

def construct_wrapped_graphcast( model_config: ModelConfig, task_config: TaskConfig, **kwargs) -> Predictor:
	gcast = graphcast.GraphCast(model_config, task_config)
	predictor: Predictor = casting.Bfloat16Cast(gcast)
#           Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from BFloat16 happens after applying normalization to the inputs/targets.
	npredictor = normalization.InputsAndResiduals( predictor, diffs_stddev_by_level=diffs_stddev_by_level, mean_by_level=mean_by_level, stddev_by_level=stddev_by_level)
#           Wraps everything so the one-step model can produce trajectories.
	apredictor = autoregressive.Predictor(npredictor, gradient_checkpointing=True)
	return apredictor

@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config)
  return predictor(inputs, targets_template=targets_template, forcings=forcings)

@hk.transform_with_state
def loss_fn(model_config, task_config, inputs, targets, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config)
  loss, diagnostics = predictor.loss(inputs, targets, forcings)
  return xarray_tree.map_structure( lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),(loss, diagnostics))

def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
  def _aux(params, state, i, t, f):
     (loss, diagnostics), next_state = loss_fn.apply( params, state, jax.random.PRNGKey(0), model_config, task_config, i, t, f)
     return loss, (diagnostics, next_state)
  (loss, (diagnostics, next_state)), grads = jax.value_and_grad( _aux, has_aux=True )(params, state, inputs, targets, forcings)
  return loss, diagnostics, next_state, grads

