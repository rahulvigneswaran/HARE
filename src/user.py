import numpy as np
import pandas as pd
import random, time
from typing import Any, List, Union
from carla.data.api import Data
from carla.models.api import MLModel
import carla.recourse_methods.catalog as recourse_catalog
from carla.recourse_methods.processing import check_counterfactuals
import torch
from sampling import sphere_packing_centers, hyperspherical_sampling
from utils import set_random_seed

'''
every user has a factual point, and a counterfactual generated (according to a seed and type in ['close', 'far'])
can query user for preferred counterfactual
'''
_GT_GEN_DISTANCE = {'near': 0.2, 'intermediate': 0.8, 'far': 1.5}

def get_preference_weights(d):
	#5 components
	# centers = [0.2, 0.5, 1, 1.5, 2]
	centers = [0.2, 1, 2, 3, 4]
	#sample d components
	mus = np.random.choice(centers, size=d)
	#sample from a multivariate guassian with this mean
	pref = np.random.multivariate_normal(mus, cov=np.eye(d))
	#truncate it accordingly
	pref = np.clip(pref, a_min=0.1, a_max=5)

	return pref

class User:
	def __init__(self, factual: pd.DataFrame, immutables: List, mlmodel: MLModel , state: int, ctype: str='far') -> None:
		self._mlmodel = mlmodel
		self._immutables = self._mlmodel.data.immutables if immutables is None else immutables
		self._raw_factual = factual.copy()
		self._factual = self._mlmodel.get_ordered_features(factual) #ordered factual df
		self._eps = _GT_GEN_DISTANCE[ctype]
		self._catp = -1
		self._num_samples = 300
		#generate counterfactual
		self._counterfactual = self._generate_gt_counterfactual(self._factual, self._mlmodel,\
		 					self._immutables, state, ctype)
		self._preferences = get_preference_weights(self._factual.to_numpy().size)

		
	@property
	def factual(self):
		return self._factual

	@property
	def counterfactual(self):
		return self._counterfactual

	@property
	def preferences(self):
		return self._preferences
	
	@property
	def immutables(self):
		return self._immutables

	@property
	def eps(self):
		return self._eps
	@property
	def catp(self):
		return self._catp
	
	@property
	def num_samples(self):
		return self._num_samples
	
	#get N samples arounf cd changing only cont features
	def _get_samples_around_cf(self, cf: pd.DataFrame, N):
		model = self._mlmodel
		immutables = self.immutables
		#N samples arounf cf in an actionable way(change only continuous features)
		encoded_categorical = model.data.encoder.get_feature_names(model.data.categorical)
		encoded_immutables = [encoded_categorical[model.data.categorical.index(name)] if name in model.data.categorical else name for name in immutables]
		categorical_indices = [cf.columns.get_loc(name) for name in encoded_categorical]
		continuous_indices = [cf.columns.get_loc(name) for name in model.data.continuous]
		immutables_indices = [cf.columns.get_loc(name) for name in encoded_immutables]
		mutable_cont_indices = list(set(continuous_indices) - set(immutables_indices))
		
		z = 2*np.random.rand(3*N + 1, len(mutable_cont_indices)) - 1
		z = (z/np.linalg.norm(z, ord=2))*(self.eps/100) #magnitude self.eps
		z[0,:] = 0 #add the current point also
		cf_samples = np.zeros((3*N+1, cf.shape[1]))
		cf_samples[:,mutable_cont_indices] = z
		cf_samples += cf.to_numpy().reshape(1,-1)

		#check validity
		cf_samples = pd.DataFrame(cf_samples, columns=cf.columns)
		cf_samples = model.get_ordered_features(check_counterfactuals(model, cf_samples, cf_samples.index))
		#get valid mask
		valid_mask = ~np.isnan(cf_samples.to_numpy()).any(axis=1)
		cf_samples = cf_samples.iloc[valid_mask]

		return cf_samples.iloc[:N]
	
	#generate counterfactual (dataframe, ordered)
	def _generate_gt_counterfactual(self, factual, model, immutables, state, ctype):
		#set random state
		set_random_seed(state)
		#STEP1: perturb factuals
		encoded_categorical = model.data.encoder.get_feature_names(model.data.categorical)
		encoded_immutables = [encoded_categorical[model.data.categorical.index(name)] if name in model.data.categorical else name for name in immutables]
		categorical_indices = [factual.columns.get_loc(name) for name in encoded_categorical]
		continuous_indices = [factual.columns.get_loc(name) for name in model.data.continuous]
		immutables_indices = [factual.columns.get_loc(name) for name in encoded_immutables]
		mutable_cont_indices = list(set(continuous_indices) - set(immutables_indices))
		mutable_cat_indices = list(set(categorical_indices) - set(immutables_indices))

		flag = True; count=0
		while flag:
			pert_factual = factual.to_numpy().reshape(-1,).copy()

			#perturb continuous features with max magnitute self._EPS
			z = 2*np.random.rand(len(mutable_cont_indices)) - 1
			z = (z/np.linalg.norm(z, ord=2))*self.eps #a random noise with mag _EPS
			pert_factual[mutable_cont_indices] += z
			pert_factual.clip(0,1,out=pert_factual)
			#perturn one categorical feature with some probability
			pert_factual[mutable_cat_indices] = np.logical_xor(pert_factual[mutable_cat_indices],\
			 (np.random.rand(len(mutable_cat_indices)) > 1-self.catp)).astype(float)
			#immmutables do not change
			pert_factual = pd.DataFrame(pert_factual.reshape(1,-1), columns=factual.columns)
			#STEP2: run recourse on perturbed sample
			# lamb = 1 if ctype=='close' else 0.1
			
			# hyperparams = {'cost_fn':lambda x,y: torch.linalg.norm((x - y).squeeze(0), ord=np.inf),\
			# 'lambda_':lamb,'loss_type':'BCE', 'binary_cat_features':True, 'clamp':True, 'immutables': immutables}
			# recourse_method = recourse_catalog.Wachter(model, hyperparams)
			print("perturbation is: ", np.linalg.norm(pert_factual.to_numpy() - factual.to_numpy(), axis=1, ord=2).item())
			hyperparams = {'seed': state+count}
			recourse_method = recourse_catalog.GrowingSpheres(model, hyperparams)
			counterfactual = recourse_method.get_counterfactuals(pert_factual)
			if np.isnan(counterfactual.to_numpy().reshape(-1,)).any():
				flag = True
				count+=1
			else:
				flag = False

		#return a valid counterfactual (dataframe, ordered)
		counterfactual.clip(0,1,inplace=True)

		return counterfactual

	#query user about two counterfactuals, return winner
	def query(self, ccf1: pd.DataFrame, ccf2: pd.DataFrame, PREF=0, NOISE_PROB=0.0):
		#cfs already ordered
		if not PREF:
			d1 = np.linalg.norm(ccf1.to_numpy().reshape(-1) - self._counterfactual.to_numpy().reshape(-1), ord=2)
			d2 = np.linalg.norm(ccf2.to_numpy().reshape(-1) - self._counterfactual.to_numpy().reshape(-1), ord=2)
			
		else:
			#use preferences to make the decision
			d1 = np.dot(self.preferences, np.absolute(ccf1.to_numpy().reshape(-1) - self.factual.to_numpy().reshape(-1)))
			d2 = np.dot(self.preferences, np.absolute(ccf2.to_numpy().reshape(-1) - self.factual.to_numpy().reshape(-1)))
			
		#noise
		decision = not(d1 < d2)
		if NOISE_PROB > 0: # Bernoulli noise
			if np.random.choice([0, 1], p=[1-NOISE_PROB, NOISE_PROB]):
					decision = not(decision)
		elif NOISE_PROB < 0: # Logistic noise
			#simulate logistic noise as 1-sigmoid(alpha.d)
			def sigmoid(alpha, beta, x):
				return 1/(1+np.exp(-alpha*(x+beta)))
			dist = np.linalg.norm(ccf1.to_numpy().reshape(-1) - ccf2.to_numpy().reshape(-1), ord=2)
			sigmoid_noise_prob = 1-sigmoid(2, 1, dist)
			# print("noise prob: ", sigmoid_noise_prob, "distance: ", dist)
			if np.random.choice([0,1], p=[1-sigmoid_noise_prob, sigmoid_noise_prob]):
				decision = not(decision)
		return int(decision)


'''
Generate candidate counterfactuals for a user
'''
def generate_candidate_counterfactuals(ind, gen_cf, model, K, hyperspherical=True) -> pd.DataFrame:
	#generate _K valid, well spread out ccfs in a sphere of radius 5*EPS centered at gen_cf TODO: 5 to 2 changed
	#TODO: how to generate gen_cf?
	#TODO: do you need LIME?
	'''
	For now, we generate counterfactual using Wachter L1 norm
	'''	
	d = gen_cf.shape[1] #dimensionality

	immutables = ind.immutables
	encoded_categorical = model.data.encoder.get_feature_names(model.data.categorical)
	encoded_immutables = [encoded_categorical[model.data.categorical.index(name)] if name in model.data.categorical else name for name in immutables]
	categorical_indices = [gen_cf.columns.get_loc(name) for name in encoded_categorical]
	immutables_indices = [gen_cf.columns.get_loc(name) for name in encoded_immutables]
	continuous_indices = [gen_cf.columns.get_loc(name) for name in model.data.continuous]
	
	#get sphere packing; get 3x points because a lot will not be valid counterfactuals
	if not hyperspherical:
		flag = True
		while flag:
			centers = sphere_packing_centers(d, 2*ind.eps, 3*K, immutables_indices) #_5 ACTUAL SAMPLING 

			gen_cf_array = gen_cf.to_numpy().reshape(-1,)
			candidates = gen_cf_array.reshape(1,-1) + centers

			#clamp b/w 0 and 1 (both discrete and continuous)
			candidates.clip(0,1,out=candidates)
			#reconstruct categorical features
			candidates[:, categorical_indices] = np.round(candidates[:, categorical_indices])
			#check counterfactual validity
			candidates = pd.DataFrame(candidates, columns=gen_cf.columns)
			candidates = model.get_ordered_features(check_counterfactuals(model, candidates, candidates.index))
			#check if there are enough
			valid_mask = ~np.isnan(candidates.to_numpy()).any(axis=1)
			print(valid_mask.sum(), K)
			flag = True if valid_mask.sum()< K else False

		candidates = candidates.iloc[valid_mask].iloc[:K]
	else:
		candidates = hyperspherical_sampling(gen_cf, model, K, d, 10, continuous_indices, categorical_indices, immutables_indices)

	#adding gen_cf also as a candidate
	candidates = pd.concat([gen_cf, candidates], axis=0).reset_index(drop=True)

	return candidates


'''
Get best candidate index for the user
'''
def get_best_candidate_index(ind, candidates, PREF, NOISE_PROB=0.0) -> int:
	n = candidates.shape[0]
	if n==1:
		return 0
	bci = ind.query(candidates.iloc[0:1], candidates.iloc[1:2],PREF=PREF, NOISE_PROB=NOISE_PROB)
	for i in range(2,n):
		j = ind.query(candidates.iloc[bci:bci+1], candidates.iloc[i:i+1],PREF=PREF, NOISE_PROB=NOISE_PROB)
		bci = (1-j)*bci + j*i
	return bci


#get similar dissimilar points
def get_similar_dissimilar_samples(ind, candidates, bci):
	#sample around best cf for similar points
	#sample in an actionable manner
	l = candidates.shape[0]
	N = ind.num_samples
	D = pd.DataFrame()
	for i in range(l):
		if i==bci:
			#generate N similar points
			S = ind._get_samples_around_cf(candidates.iloc[i:i+1], N)
		else:
			D_ = ind._get_samples_around_cf(candidates.iloc[i:i+1], int(np.ceil(N/(l-1))))
			D = pd.concat((D, D_), axis=0, copy=False)

	#merge and return
	cf_samples_data = pd.concat((ind.factual, S, D), axis=0)
	labels = np.ones(1+S.shape[0]+D.shape[0])
	labels[-D.shape[0]:] = 0
	cf_samples_labels = pd.DataFrame(labels.astype(np.int), columns=['target'])
	
	return cf_samples_data, cf_samples_labels
