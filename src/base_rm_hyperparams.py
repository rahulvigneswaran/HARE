import torch

def get_base_rm_hyperparam(args):
    rm_type = args.recourse_method
    assert rm_type in ['wachter', 'face', 'growing_spheres', 'cchvae', 'ar', 'roar', 'crud','clue'], f"Recourse method not available: {rm_type}"
    hyperparams = {}
    # Working recourse methods
    if rm_type == 'wachter':
        hyperparams = {
            'cost_fn': lambda x, y: torch.linalg.norm((x - y).squeeze(0), ord=1),
            'lambda_': 0.1,
            'loss_type': 'BCE',
            'binary_cat_features': True,
            'clamp': True,
            'seed': args.seed,
        }
    elif rm_type == 'face':
        hyperparams = {
            'mode': 'knn',
            'fraction': 0.1,
            'p_norm': 1,
            'seed': args.seed,
        }
    elif rm_type == 'growing_spheres':
        hyperparams = {
            'seed': args.seed,
                       }
        
    elif rm_type == 'cchvae':
        if args.dataset in ["adult"]:
           hyperparams = {
            'seed': args.seed,
            "data_name": args.dataset,
            "n_search_samples": 300,
            "p_norm": 1,
            "step": 0.1,
            "max_iter": 1000,
            "clamp": True,
            "binary_cat_features": True,
            "vae_params": {
                "layers": [7,32,16,4],
                "train": True,
                "lambda_reg": 1e-6,
                "epochs": 5,
                "lr": 1e-3,
                "batch_size": 32,
            }
           }
        elif args.dataset in ["give_me_some_credit"]:
           hyperparams = {
            'seed': args.seed,
            "data_name": args.dataset,
            "n_search_samples": 300,
            "p_norm": 1,
            "step": 0.1,
            "max_iter": 1000,
            "clamp": True,
            "binary_cat_features": True,
            "vae_params": {
                "layers": [8,32,16,4],
                "train": True,
                "lambda_reg": 1e-6,
                "epochs": 5,
                "lr": 1e-3,
                "batch_size": 32,
            }
           }
        elif args.dataset == "compas":
           hyperparams = {
            'seed': args.seed,
            "data_name": args.dataset,
            "n_search_samples": 100,
            "p_norm": 1,
            "step": 0.1,
            "max_iter": 1000,
            "clamp": True,
            "binary_cat_features": True,
            "vae_params": {
                "layers": [4, 512, 256, 8 ],
                "train": True,
                "lambda_reg": 1e-6,
                "epochs": 100,
                "lr": 1e-3,
                "batch_size": 32,
            }
           }  
        elif args.dataset == "heloc":
           hyperparams = {
            'seed': args.seed,
            "data_name": args.dataset,
            "n_search_samples": 300,
            "p_norm": 1,
            "step": 0.1,
            "max_iter": 1000,
            "clamp": True,
            "binary_cat_features": True,
            "vae_params": {
                "layers": [5,32,16,4],
                "train": True,
                "lambda_reg": 1e-6,
                "epochs": 5,
                "lr": 1e-3,
                "batch_size": 32,
            }
           }    
        else:
            print(f"WARNING: Wrong dataset - {args.dataset}")
            exit()
    #---
    # Trying
    elif rm_type == 'dice':
        if args.dataset == 'adult':
            hyperparams = {
                "num": 1,
                "desired_class": 1,
                "posthoc_sparsity_param": 0.1,
                'seed': args.seed,
            }
        elif args.dataset == 'compas':
            hyperparams = {
                "num": 1,
                "desired_class": 0,
                "posthoc_sparsity_param": 0.1,
                'seed': args.seed,
            }
        elif args.dataset == 'heloc':
            hyperparams = {
                "num": 1,
                "desired_class": 1,
                "posthoc_sparsity_param": 0.1,
                'seed': args.seed,
            }
        else:
            print(f"WARNING: Wrong dataset - {args.dataset}")
            exit()
        # hyperparams = {
        #     "num": 1,
        #     "desired_class": 1,
        #     "posthoc_sparsity_param": 0.1,
        #     'seed': args.seed,
        # }
    elif rm_type == 'clue':
        if args.dataset == 'adult':
            hyperparams =                  {
                            "data_name": args.dataset,
                            'seed': args.seed,
                            "train_vae": True,
                            "width": 10,
                            "depth": 3,
                            "latent_dim": 12,
                            "batch_size": 64,
                            "epochs": 5,
                            "lr": 0.001,
                            "early_stop": 10,
                        }
        elif args.dataset == "give_me_some_credit":
            hyperparams =                  {
                            "data_name": args.dataset,
                            'seed': args.seed,
                            "train_vae": True,
                            "width": 10,
                            "depth": 3,
                            "latent_dim": 12,
                            "batch_size": 64,
                            "epochs": 5,
                            "lr": 0.001,
                            "early_stop": 10,
                        }
        elif args.dataset == 'compas':
            hyperparams =                        {
                            "data_name": args.dataset,
                            'seed': args.seed,
                            "train_vae": True,
                            "width": 10,
                            "depth": 3,
                            "latent_dim": 12,
                            "batch_size": 64,
                            "epochs": 5,
                            "lr": 0.01,
                            "early_stop": 10,
                        }
        elif args.dataset == 'heloc':
            hyperparams =                        {
                            "data_name": args.dataset,
                            'seed': args.seed,
                            "train_vae": True,
                            "width": 10,
                            "depth": 3,
                            "latent_dim": 12,
                            "batch_size": 64,
                            "epochs": 5,
                            "lr": 0.001,
                            "early_stop": 10,
                        }
        else:
            print(f"WARNING: Wrong dataset - {args.dataset}")
            exit()
    
    elif rm_type == 'crud':
        if args.dataset == "adult":
            hyperparams =                         {
                            "data_name": args.dataset,
                            'seed': args.seed,
                            "target_class": [0, 1],
                            "lambda_param": 0.001,
                            "optimizer": "RMSprop",
                            "lr": 0.008,
                            "max_iter": 2000,
                            "binary_cat_features": False,
                            "vae_params": {
                                "layers": [7,32,16,4],
                                "train": True,
                                "epochs": 5,
                                "lr": 1e-3,
                                "batch_size": 1024,
                            }
                        }
        elif args.dataset == "give_me_some_credit":
            hyperparams =                         {
                    "data_name": args.dataset,
                    'seed': args.seed,
                    "target_class": [0, 1],
                    "lambda_param": 0.001,
                    "optimizer": "RMSprop",
                    "lr": 0.008,
                    "max_iter": 2000,
                    "binary_cat_features": False,
                    "vae_params": {
                        "layers": [8,32,16,4],
                        "train": True,
                        "epochs": 5,
                        "lr": 1e-3,
                        "batch_size": 1024,
                    }
                }
        elif args.dataset == 'compas':
            hyperparams =                        {
                            "data_name": args.dataset,
                            'seed': args.seed,
                            "target_class": [0, 1],
                            "lambda_param": 0.001,
                            "optimizer": "RMSprop",
                            "lr": 0.008,
                            "max_iter": 2000,
                            "binary_cat_features": False,
                            "vae_params": {
                                "layers": [4, 512, 256, 8 ],
                                "train": True,
                                "epochs": 5,
                                "lr": 1e-2,
                                "batch_size": 1024,
                            }
                        }
        elif args.dataset == 'heloc':
            hyperparams =                        {
                            "data_name": args.dataset,
                            'seed': args.seed,
                            "target_class": [0, 1],
                            "lambda_param": 0.001,
                            "optimizer": "RMSprop",
                            "lr": 0.008,
                            "max_iter": 2000,
                            "binary_cat_features": False,
                            "vae_params": {
                                "layers": [5,32,16,4],
                                "train": True,
                                "epochs": 5,
                                "lr": 1e-3,
                                "batch_size": 1024,
                            }
                        }
        else:
            print(f"WARNING: Wrong dataset - {args.dataset}")
            exit()
    #--
    # Not tried yet
    elif rm_type == 'roar':
        hyperparams = {
            "feature_cost": "_optional_",
            "lr": 0.01,
            "lambda_": 0.01,
            "delta_max": 0.01,
            "norm": 1,
            "t_max_min": 0.5,
            "loss_type": "BCE",
            "y_target": [0, 1],
            "binary_cat_features": True,
            "loss_threshold": 1e-3,
            "discretize": False,
            "sample": True,
            "lime_seed": 0,
            "seed": args.seed,
        }

    elif rm_type == 'ar':
        hyperparams = {
            "fs_size": 100,
            "discretize": False,
            "sample": True,
            'seed': args.seed,
        }

    return hyperparams

# def get_base_rm_hyperparam_sweep(args):
#     rm_type = args.recourse_method
#     assert rm_type in ['wachter', 'face', 'growing_spheres', 'cchvae', 'ar', 'roar', 'dice'], f"Recourse method not available: {rm_type}"
#     hyperparams = {}

#     if rm_type == 'wachter':
#         hyperparams = {
#             'cost_fn': lambda x, y: torch.linalg.norm((x - y).squeeze(0), ord=1),
#             'lambda_': 0.1,
#             'loss_type': 'BCE',
#             'binary_cat_features': True,
#             'clamp': True,
#             'seed': args.seed,
#         }
#     elif rm_type == 'face':
#         hyperparams = {
#             'mode': 'knn',
#             'fraction': 0.1,
#             'p_norm': 1,
#             'seed': args.seed,
#         }
#     elif rm_type == 'growing_spheres':
#         hyperparams = {
#             'seed': args.seed,
#                        }
#     elif rm_type == 'cchvae':
#         if args.dataset == "adult":
#            hyperparams = {
#             'seed': args.seed,
#             "data_name": args.dataset,
#             "n_search_samples": 300,
#             "p_norm": 1,
#             "step": 0.1,
#             "max_iter": 1000,
#             "clamp": True,
#             "binary_cat_features": True,
#             "vae_params": {
#                 "layers": [20,32,16,4],
#                 "train": True,
#                 # "kl_weight": args.arg4,
#                 "lambda_reg": 1e-6,
#                 "epochs": 5,
#                 "lr": 1e-3,
#                 "batch_size": 32,
#             }
#            }
#         elif args.dataset == "compas":
#            hyperparams = {
#             'seed': args.seed,
#             "data_name": args.dataset,
#             "n_search_samples": 100,
#             "p_norm": 1,
#             "step": 0.1,
#             "max_iter": 1000,
#             "clamp": True,
#             "binary_cat_features": True,
#             "vae_params": {
#                 "layers": [10, 512, 256, 8 ],
#                 "train": True,
#                 # "kl_weight": args.arg4,
#                 "lambda_reg": 1e-6,
#                 "epochs": 100,
#                 "lr": 1e-3,
#                 "batch_size": 32,
#             }
#            }  
#         elif args.dataset == "heloc":
#            hyperparams = {
#             'seed': args.seed,
#             "data_name": args.dataset,
#             "n_search_samples": 300,
#             "p_norm": 1,
#             "step": 0.1,
#             "max_iter": 1000,
#             "clamp": True,
#             "binary_cat_features": True,
#             "vae_params": {
#                 "layers": [21,32,16,4],
#                 "train": True,
#                 # "kl_weight": args.arg4,
#                 "lambda_reg": 1e-6,
#                 "epochs": 5,
#                 "lr": 1e-3,
#                 "batch_size": 32,
#             }
#            }    
#         else:
#             print(f"WARNING: Wrong dataset - {args.dataset}")
#             exit()
#     elif rm_type == 'roar':
#         hyperparams = {
#             "feature_cost": "_optional_",
#             "lr": 0.01,
#             "lambda_": 0.01,
#             "delta_max": 0.01,
#             "norm": 1,
#             "t_max_min": 0.5,
#             "loss_type": "BCE",
#             "y_target": [0, 1],
#             "binary_cat_features": True,
#             "loss_threshold": 1e-3,
#             "discretize": False,
#             "sample": True,
#             "lime_seed": 0,
#             "seed": args.seed,
#         }
#     elif rm_type == 'ar':
#         hyperparams = {
#             "fs_size": args.arg1,
#             "discretize": False,
#             "sample": True,
#             'seed': args.seed,
#         }
#     elif rm_type == 'dice':
#         if args.dataset == 'adult':
#             hyperparams = {
#                 "num": 1,
#                 "desired_class": 1,
#                 "posthoc_sparsity_param": 0.1,
#                 'seed': args.seed,
#             }
#         elif args.dataset == 'compas':
#             hyperparams = {
#                 "num": 1,
#                 "desired_class": 1,
#                 "posthoc_sparsity_param": 0.1,
#                 'seed': args.seed,
#             }

#     return hyperparams
