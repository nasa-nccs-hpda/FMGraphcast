from fmbase.source.merra2.model import MERRA2DataInterface
from graphcast.graphcast import ModelConfig, TaskConfig, GraphCast
from graphcast import casting, normalization
from graphcast.autoregressive import Predictor
from fmbase.util.config import cfg
from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree
from IPython.display import HTML
import ipywidgets as widgets
import haiku as hk
import jax
import functools

class FMGraphcast(MERRA2DataInterface):

	def __int__(self, **kwargs):
		MERRA2DataInterface.__init__(self )
		self.model_config: ModelConfig = self.config_model( **kwargs )
		self.task_config: TaskConfig = self.config_task(**kwargs)
		self.predictor: Predictor = self.construct_wrapped_graphcast(**kwargs)

	@classmethod
	def config_model(cls, **kwargs ) -> ModelConfig:
		mc = ModelConfig()
		mc.resolution=     kwargs.get('resolution',    cfg().model.resolution),
		mc.mesh_size=      kwargs.get('mesh_size',     cfg().model.mesh_size),
		mc.latent_size=    kwargs.get('latent_size',   cfg().model.latent_size),
		mc.gnn_msg_steps=  kwargs.get('gnn_msg_steps', cfg().model.gnn_msg_steps),
		mc.hidden_layers=  kwargs.get('hidden_layers', cfg().model.hidden_layers),
		mc.radius_query_fraction_edge_length= kwargs.get('radius_query_fraction_edge_length', cfg().model.radius_query_fraction_edge_length)
		return mc

	@classmethod
	def config_task(cls, **kwargs) -> TaskConfig:
		tc =  TaskConfig()
		tc.input_variables=    kwargs.get('input_variables',    cfg().task.input_variables)
		tc.target_variables=   kwargs.get('target_variables',   cfg().task.target_variables)
		tc.forcing_variables=  kwargs.get('forcing_variables',  cfg().task.forcing_variables)
		tc.pressure_levels=    kwargs.get('z_levels',           cfg().task.z_levels)
		tc.input_duration=     kwargs.get('input_duration',     cfg().task.input_duration)
		return tc

	def construct_wrapped_graphcast( self, **kwargs ) -> Predictor:

		predictor = GraphCast(self.model_config, self.task_config)

		# Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
		# from/to float32 to/from BFloat16.
		predictor = casting.Bfloat16Cast(predictor)

		# # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
		# # BFloat16 happens after applying normalization to the inputs/targets.
		# predictor = normalization.InputsAndResiduals(
		# 	predictor,
		# 	diffs_stddev_by_level=diffs_stddev_by_level,
		# 	mean_by_level=mean_by_level,
		# 	stddev_by_level=stddev_by_level)

		predictor = Predictor(predictor, gradient_checkpointing=True)
		return predictor

	@hk.transform_with_state
	def run_forward(self, inputs, targets_template, forcings):
		predictor = self.construct_wrapped_graphcast()
		return predictor(inputs, targets_template=targets_template, forcings=forcings)

	@hk.transform_with_state
	def loss_fn(self, inputs, targets, forcings):
		predictor = self.construct_wrapped_graphcast()
		loss, diagnostics = predictor.loss(inputs, targets, forcings)
		return xarray_tree.map_structure(
			lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
			(loss, diagnostics))

	def grads_fn(self, params, state, loss_fn, inputs, targets, forcings):
		def _aux(params, state, i, t, f):
			(loss, diagnostics), next_state = loss_fn.apply( params, state, jax.random.PRNGKey(0), i, t, f)
			return loss, (diagnostics, next_state)

		(loss, (diagnostics, next_state)), grads = jax.value_and_grad( _aux, has_aux=True)(params, state, inputs, targets, forcings)
		return loss, diagnostics, next_state, grads

	# Jax doesn't seem to like passing configs as args through the jit. Passing it
	# in via partial (instead of capture by closure) forces jax to invalidate the
	# jit cache if you change configs.
	def with_configs(fn):
		return functools.partial(
			fn, model_config=model_config, task_config=task_config)

	# Always pass params and state, so the usage below are simpler
	def with_params(fn):
		return functools.partial(fn, params=params, state=state)

	# Our models aren't stateful, so the state is always empty, so just return the
	# predictions. This is requiredy by our rollout code, and generally simpler.
	def drop_state(fn):
		return lambda **kw: fn(**kw)[0]

	init_jitted = jax.jit(with_configs(run_forward.init))

	if params is None:
		params, state = init_jitted(
			rng=jax.random.PRNGKey(0),
			inputs=train_inputs,
			targets_template=train_targets,
			forcings=train_forcings)

	loss_fn_jitted = drop_state(with_params(jax.jit(with_configs(loss_fn.apply))))
	grads_fn_jitted = with_params(jax.jit(with_configs(grads_fn)))
	run_forward_jitted = drop_state(with_params(jax.jit(with_configs(run_forward.apply))))
