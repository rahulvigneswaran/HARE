'''
Config file
TODO: have a default config yaml and merge cli
'''
import argparse

class config:
	parser = argparse.ArgumentParser()
	#dataset and base model
	parser.add_argument('--dataset', choices=['adult', 'give_me_some_credit', 'compas', 'heloc'],\
	 	default='adult', help='dataset choice')
	parser.add_argument('--mlmodel', choices=['linear', 'ann'], default='ann', help='base model type')
	parser.add_argument('--retrain', action='store_true', default=True, help='flag to force base model retraining')
	#recourse method
	parser.add_argument('--recourse_method', choices=['wachter', 'face', 'growing_spheres', 'cchvae','dice', 'clue', 'crud' ,'ar', 'roar',], default='wachter', help='choice of base recourse generator')
	parser.add_argument('--our_recourse_method', choices=['binary_search', 'binary_search_multi_iter','None', 'randumb'], default='binary_search_multi_iter', help='choice of recourse generator for ++')
	# parser.add_argument('--norm', type=int, default=1,\
	# 	help='which norm to use as cost function to generate recourse')
	parser.add_argument('--lamb', type=float, default=0.1,\
		help='regularization parameter for cost')
	parser.add_argument('--K', type=int, default=30,\
		help='number of candidate counterfactuals')
	parser.add_argument('--gt_perturbation', type=str, default='far', help='radius of gt perturbation ball')
	# parser.add_argument('--bs_final_only', action='store_true', default=False, help='By default we do bs to all candidates. This will force bs to be done only to the final candidate.')
	parser.add_argument('--bs_final_only', type=int, default=0, help='By default we do bs to all candidates. This will force bs to be done only to the final candidate.')
	parser.add_argument('--bs_final_onlyv2', type=int, default=0, help='By default we do bs to all candidates. This will force bs to be done only to the final candidate.')
	parser.add_argument('--multi_iter', type=int, default=5, help='iterations for multi_iter [3,5,6,10] for K = 30')
	parser.add_argument('--noise_prob', type=float, default=0.0, help='Noisy User. Probability with which the user decision is swapped to make them noise. Options: > 0.0 - Bernoulli Noise, < -1 - Logistic Noise, 0.0 - No Noise')
	parser.add_argument('--use_prefs', type=int, default=0, help='Runs the preferences setup')

	#other args
	parser.add_argument("--gpu", default="7", type=str, help="Select the GPU  to be used.")
	parser.add_argument('--testsize', type=int, default=100, help='number of test points')
	parser.add_argument('--seed', default=1, type=int, help='mandatory seed')
	parser.add_argument('--experiment', default='',\
	 	help='experiment description to be appened to output file')
	parser.add_argument('--stdout', action='store_true',\
		help='if true write to stdout, else to a file')
	
	args = parser.parse_args()

	@classmethod
	def cli_args(cls):
		return cls.args
