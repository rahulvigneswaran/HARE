'''
Main function
'''
import wandb
import warnings
warnings.filterwarnings('ignore')

from tqdm import trange
import os, sys, time, random, math
from pathlib import Path
import argparse
from tabulate import tabulate
import pdb

import numpy as np
import pandas as pd
import torch
import copy 

from carla.data.catalog import OnlineCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
import carla.recourse_methods.catalog as recourse_catalog
import carla.evaluation.catalog as evaluation_catalog

from user import *
from utils import *
from config import config

from base_rm_hyperparams import *

'''
Function to generate counterfactuals
'''
def _get_recourse_method(model, rm_type, **kwargs):
	hyperparams = get_base_rm_hyperparam(args)
	if rm_type=='wachter':
		method = recourse_catalog.Wachter(model, hyperparams)
	elif rm_type=='face':
		method = recourse_catalog.Face(model, hyperparams)
	elif rm_type=='growing_spheres':
		method = recourse_catalog.GrowingSpheres(model, hyperparams)
	elif rm_type=='cchvae':
		method = recourse_catalog.CCHVAE(model, hyperparams)
	elif rm_type=='ar':
		method = recourse_catalog.ActionableRecourse(model, hyperparams)
	elif rm_type=='roar':
		method = recourse_catalog.Roar(model, hyperparams)
	elif rm_type=='clue':
		method = recourse_catalog.Clue(model.data.df[model.feature_input_order + [model.data.target]], model, hyperparams)
	elif rm_type=='crud':
		method = recourse_catalog.CRUD(model, hyperparams)
	elif rm_type=='dice':
		method = recourse_catalog.Dice(model, hyperparams)
	elif rm_type=='fare':
		method = recourse_catalog.FARE(model)

	return method

def _generate_cf(u, model, base_rm, args):
	print(f"----------- Generating {args.recourse_method} counterfactual -------------")
	pdb.set_trace()
	if args.recourse_method == "crud": 	
		cf = base_rm.get_counterfactuals(u._raw_factual)
	else:
		cf = base_rm.get_counterfactuals(u.factual)
	return cf

def _generate_binsearch_cf(u, model, base_rm, args):
	print(f"----------- Generating init {args.recourse_method} counterfactual -------------")
	if args.recourse_method == "crud": 
		init_cf = base_rm.get_counterfactuals(u._raw_factual)
	else:
		init_cf = base_rm.get_counterfactuals(u.factual)
	if np.isnan(init_cf.to_numpy()).any():
		print("cf is nan!")
		wandb.log({"init_cf_nan": True})
		return pd.DataFrame(init_cf.to_numpy(), columns=u.factual.columns, index=u.factual.index)
	
	#generate candidates
	candidates = generate_candidate_counterfactuals(u, init_cf, model, args.K, hyperspherical=True) #SAMPLING CFS
	if not(args.bs_final_only):
		candidates = np.concatenate([get_binary_search_cf(u.factual, candidates.iloc[i:i+1], model, 1e-06) for i in range(candidates.shape[0])], axis=0) #binsearch candidates
	candidates = pd.DataFrame(candidates, columns=u.factual.columns).reset_index(drop=True)

	#get best candidate index
	bci = get_best_candidate_index(u, candidates, PREF=args.use_prefs) #GETTING BEST CF FROM THE USER

	print(f"----------- Generating final {args.recourse_method} counterfactual -------------")
	if args.bs_final_only:
		cf = get_binary_search_cf(u.factual, candidates.iloc[bci:bci+1], model, 1e-06) #RUN OUR ALGO USING BEST CF
	else:
		cf = pd.DataFrame(candidates.iloc[bci:bci+1].to_numpy(), columns=u.factual.columns, index=u.factual.index)
	return cf

def _generate_randumb_cf(u, model, base_rm, args):
	print(f"----------- Generating init {args.recourse_method} counterfactual -------------")
	if args.recourse_method == "crud": 
		init_cf = base_rm.get_counterfactuals(u._raw_factual)
	else:
		init_cf = base_rm.get_counterfactuals(u.factual)
	if np.isnan(init_cf.to_numpy()).any():
		print("cf is nan!")
		wandb.log({"init_cf_nan": True})
		return pd.DataFrame(init_cf.to_numpy(), columns=u.factual.columns, index=u.factual.index)
	
	#generate candidates
	candidates = generate_candidate_counterfactuals(u, init_cf, model, args.K, hyperspherical=True) #SAMPLING CFS
	if not(args.bs_final_only):
		candidates = np.concatenate([get_binary_search_cf(u.factual, candidates.iloc[i:i+1], model, 1e-06) for i in range(candidates.shape[0])], axis=0) #binsearch candidates
	candidates = pd.DataFrame(candidates, columns=u.factual.columns).reset_index(drop=True)

	# Simple random selection
	import random
	bci = random.randint(0, len(candidates)-1)
	
	print(f"----------- Generating final {args.recourse_method} counterfactual -------------")
	if args.bs_final_only:
		cf = get_binary_search_cf(u.factual, candidates.iloc[bci:bci+1], model, 1e-06) #RUN OUR ALGO USING BEST CF
	else:
		cf = pd.DataFrame(candidates.iloc[bci:bci+1].to_numpy(), columns=u.factual.columns, index=u.factual.index)
	return cf

def _generate_binsearch_cf_recursive(u, model, base_rm, args):
	print(f"----------- Generating init {args.recourse_method} counterfactual -------------")
	if args.recourse_method == "crud": 	
		init_cf = base_rm.get_counterfactuals(u._raw_factual)
	else:
		init_cf = base_rm.get_counterfactuals(u.factual)
	if np.isnan(init_cf.to_numpy()).any():
		print("cf is nan!")
		wandb.log({"init_cf_nan": True})
		return pd.DataFrame(init_cf.to_numpy(), columns=u.factual.columns, index=u.factual.index)
	
	# recursively choose best candidates
	div_by = args.multi_iter
	best_cf = init_cf # at start the best_cf is the initial counterfactual
	for round in range(args.K//div_by): #assumes args.K%div_by == 0
		#generate candidates
		print(f"CF generation round: {round+1}/{args.K//div_by}")
		candidates = generate_candidate_counterfactuals(u, best_cf, model, div_by) #SAMPLING CFS 
		if not(args.bs_final_only):
			candidates = np.concatenate([get_binary_search_cf(u.factual, candidates.iloc[i:i+1], model, 1e-06) for i in range(candidates.shape[0])], axis=0) #binsearch candidates
		candidates = pd.DataFrame(candidates, columns=u.factual.columns).reset_index(drop=True)
		
		#get best candidate index
		bci = get_best_candidate_index(u, candidates, PREF=args.use_prefs, NOISE_PROB=args.noise_prob) #GETTING BEST CF FROM THE USER
	
		if args.bs_final_only:
			best_cf = get_binary_search_cf(u.factual, candidates.iloc[bci:bci+1], model, 1e-06) #RUN OUR ALGO USING BEST CF
		else:
			best_cf = pd.DataFrame(candidates.iloc[bci:bci+1].to_numpy(), columns=u.factual.columns, index=u.factual.index)
		
	return best_cf

##################################################################################
# MAIN
##################################################################################
if __name__=='__main__':

	use_wandb = False
	args = config.cli_args()
	if use_wandb:
		wandb.init(project="<your_project_name>", config=args, 
					name=None if args.experiment == '' else args.experiment,
					save_code=True,
					mode="online"
					)
	
	set_random_seed(args.seed)

	#device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	#Path
	basepath = Path.cwd().parents[0]

	#immutables are set in data_catalog.yaml
	dataset = OnlineCatalog(args.dataset)
	#cache model
	model = MLModelCatalog(dataset, args.mlmodel, 'pytorch', cache=True,\
	 	models_home= basepath / 'model_cache', load_online=False)
	train_kwargs = {'learning_rate':2e-3, 'epochs':20, 'force_train':args.retrain,\
	 	'batch_size':256, 'hidden_size': [10,5,10]}
	model.train(**train_kwargs)

	#get factuals
	factuals = predict_negative_instances(model, model.data.df_test).reset_index(drop=True).iloc[:args.testsize]
	
	assert len(factuals) > 0, "No Factuals!"


	#create individuals
	num_users = factuals.shape[0]
	userlist = []
	gen_cfs = pd.DataFrame()	
	gt_cfs = pd.DataFrame()
	prefs_vector = np.array([]).reshape(0, model.get_ordered_features(factuals).shape[1])
	base_rm = _get_recourse_method(model, args.recourse_method)
	for i in trange(num_users):
		print("\n----------- creating individual -----------")
		u = User(factuals.iloc[i:i+1], None, model, args.seed, args.gt_perturbation)
		gt_cfs = pd.concat((gt_cfs, u.counterfactual), axis=0, copy=False)
		# pdb.set_trace()
		prefs_vector = np.vstack((prefs_vector, u.preferences.reshape(1,-1))) 

		print("----------- Individual created ------------")
		
		#generate cf for each individual
		if args.our_recourse_method=='binary_search':
			cf = _generate_binsearch_cf(u, model, copy.deepcopy(base_rm), args)
		elif args.our_recourse_method=='binary_search_multi_iter':
			cf = _generate_binsearch_cf_recursive(u, model, copy.deepcopy(base_rm), args)
		elif args.our_recourse_method=='randumb':
			cf = _generate_randumb_cf(u, model, copy.deepcopy(base_rm), args)
		else:
			cf = _generate_cf(u, model, copy.deepcopy(base_rm), args)

		gen_cfs = pd.concat((gen_cfs, cf), axis=0, copy=False)
		userlist.append(u)
	
	assert gen_cfs.shape[0]==len(userlist)==factuals.shape[0]
	factuals = model.get_ordered_features(factuals)

	evaluation_measures = [
		evaluation_catalog.Distance(model),
		evaluation_catalog.SuccessRate(),
		evaluation_catalog.ConstraintViolation(model),
		evaluation_catalog.Redundancy(model, {'cf_label':1}),
	]
	pipeline = [pd.DataFrame([measure.get_evaluation(factuals=factuals, counterfactuals=gen_cfs).mean(axis=0)]) for measure in evaluation_measures]
	output = pd.concat(pipeline, axis=1)

	#compute proximity between gtg-cf and generated counterfactuals
	gt_proximity = gt_cost(factuals, gt_cfs, gen_cfs)
	gt_proximity = pd.DataFrame([[gt_proximity['L2'][0], gt_proximity['cosine'][0]]], columns=['gt_L2_distance', 'gt_cosine_distance'])
	prefs_proximity = pref_cost(factuals, prefs_vector, gen_cfs)
	prefs_proximity = pd.DataFrame([[prefs_proximity['factual_L2'][0], prefs_proximity['factual_L1'][0], prefs_proximity['preference'][0]]], columns=['factual_L2', 'factual_L1', 'factual_preferences'])
	output = pd.concat([gt_proximity, output], axis=1)
	output = pd.concat([prefs_proximity, output], axis=1)
	
	#hardcode the way we want to visualize the output
	cols = ['Success_Rate', 'gt_L2_distance', 'gt_cosine_distance', 'Constraint_Violation', 'Redundancy', 'L2_distance', 'L0_distance']
	if args.use_prefs:
		cols = ['factual_L2','factual_preferences'] + cols

	print(tabulate(output[cols], headers='keys', tablefmt='psql', showindex=False))

	# wandb logging
	if use_wandb:
		wandb.log({key: value[0] for key, value in output.to_dict().items()})

	custom_path = "logs"
	if not(Path(custom_path).is_dir()) : os.mkdir(custom_path)
	bs_flag = "_bs_final_only" if args.bs_final_only else ""
	if args.noise_prob == 0.0:
		fname = f'./{custom_path}/{args.recourse_method}_{args.our_recourse_method}_{args.dataset}_{args.mlmodel}_{args.gt_perturbation}_{args.seed}{bs_flag}_K_{args.K}_iter_{args.multi_iter}.txt'
	else:
		fname = f'./{custom_path}/{args.recourse_method}_{args.our_recourse_method}_{args.dataset}_{args.mlmodel}_{args.gt_perturbation}_{args.seed}{bs_flag}_K_{args.K}_iter_{args.multi_iter}_logisticnoise_alpha2_beta1.txt'

	if args.use_prefs:
		fname = f'./{custom_path}/{args.recourse_method}_{args.our_recourse_method}_{args.dataset}_{args.mlmodel}_{args.gt_perturbation}_{args.seed}{bs_flag}_K_{args.K}_iter_{args.multi_iter}_prefs_{args.use_prefs}.txt'

	if Path(fname).is_file():
		new_file_flag = False
		modifier = 'a'
	else:
		new_file_flag = True
		modifier = 'w'
	with open(fname, modifier) as f:
		if new_file_flag:
			f.write(str(cols)+'\n')
		f.write(f'seed {args.seed}\n')
		f.write(', '.join(str(output[cols].to_numpy().reshape(-1))[1:-1].split()) +'\n')

	if use_wandb:
		wandb.finish()
