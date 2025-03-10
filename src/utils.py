import random
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cosine as scipy_cosine

def set_random_seed(mseed):
	random.seed(mseed)
	np.random.seed(mseed)
	torch.manual_seed(mseed)
	torch.cuda.manual_seed_all(mseed)

	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def gt_cost(factuals, gt_cfs, gen_cfs, nan_sensitive=False):
	#the set of factuals is the same for both
	gt_cfs = gt_cfs.to_numpy()
	gen_cfs = gen_cfs.to_numpy()
	factuals = factuals.to_numpy()
	nanmask1 = np.isnan(gen_cfs).any(axis=1)
	nanmask2 = np.isnan(gt_cfs).any(axis=1)
	nanmask = np.logical_or(nanmask1, nanmask2)
	if nan_sensitive:
		gen_cfs[nanmask] = 0.0
	else:
		gt_cfs = gt_cfs[~nanmask]
		gen_cfs = gen_cfs[~nanmask]
		factuals = factuals[~nanmask]
	cosine = np.array([scipy_cosine(gt_cfs[i]-factuals[i], gen_cfs[i]-factuals[i]) for i in range(len(gt_cfs))])
	l1_cost = np.linalg.norm(gt_cfs - gen_cfs, ord=1, axis=1)
	l2_cost = np.linalg.norm(gt_cfs - gen_cfs, ord=2, axis=1)
	return {'L1': (l1_cost.mean().item(), l1_cost.std().item()), 'L2': (l2_cost.mean().item(), l2_cost.std().item()),\
	'cosine': (cosine.mean().item(), cosine.std().item())}

def pref_cost(factuals, preferences, gen_cfs, nan_sensitive=False):
	#the set of factuals is the same for both
	gen_cfs = gen_cfs.to_numpy()
	factuals = factuals.to_numpy()
	nanmask = np.isnan(gen_cfs).any(axis=1)
	if nan_sensitive:
		gen_cfs[nanmask] = 0.0
	else:
		gen_cfs = gen_cfs[~nanmask]
		factuals = factuals[~nanmask]
		preferences = preferences[~nanmask]
	differences = gen_cfs - factuals
	preference_costs = (preferences * np.absolute(differences)).sum(axis=1)
	
	# Calculate additional metrics
	l1_cost = np.linalg.norm(differences, ord=1, axis=1)
	l2_cost = np.linalg.norm(differences, ord=2, axis=1)
	
	return {'factual_L1': (l1_cost.mean().item(), l1_cost.std().item()), 'factual_L2': (l2_cost.mean().item(), l2_cost.std().item()),\
	'preference': (preference_costs.mean().item(), preference_costs.std().item())}


def get_binary_search_cf(factual, counterfactual, model, eps):
	#the cf was generated using immutability constraints
	start = factual.to_numpy().reshape(-1)
	end = counterfactual.to_numpy().reshape(-1)
	assert not np.isnan(end).any(), 'candidate cf has nans'

	while(np.linalg.norm(start-end, ord=2) > eps):
		#break if within tolerance
		mid = (start + end)/2
		
		if model.predict(mid.reshape(1,-1))>=0.5:
			#mid lies in positive region
			end = mid
		else:
			start = mid

	gen_cf = pd.DataFrame(end.reshape(1,-1), columns=factual.columns, index=factual.index) 
	
	return gen_cf