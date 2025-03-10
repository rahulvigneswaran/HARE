import warnings
warnings.filterwarnings('ignore')
from carla.data.catalog import OnlineCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
import carla.recourse_methods.catalog as recourse_catalog

import numpy as np
import random

if __name__=='__main__':

	#set random seed
	mseed = 1
	random.seed(mseed)
	np.random.seed(mseed)

	dataset = OnlineCatalog('adult', scaling_method='Standard', encoding_method='OneHot_drop_binary')
	mlmodel = MLModelCatalog(dataset, 'ann', 'pytorch', models_home='/raid/ksrinivas/projects/CARLA/model_cache', load_online=False)
	train_kwargs = {'learning_rate':2e-3, 'epochs':10, 'force_train':False,\
	 	'batch_size':256, 'hidden_size': [10,5,10]}
	mlmodel.train(**train_kwargs)

	factuals = predict_negative_instances(mlmodel, mlmodel.data.df_test).reset_index(drop=True)
	factuals = mlmodel.get_ordered_features(factuals.iloc[:10])

	hyperparams = {'seed': mseed}
	gs = recourse_catalog.GrowingSpheres(mlmodel, hyperparams)
	#why is growing spheres not deterministic??

	cfs = gs.get_counterfactuals(factuals)

	print(cfs.to_string())

	