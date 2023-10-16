from fmbase.source.merra2.model import MERRA2DataInterface
from graphcast.graphcast import ModelConfig, TaskConfig, GraphCast
from graphcast import casting, normalization
from graphcast.autoregressive import Predictor
from fmbase.util.config import cfg

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
