'''
Code for deep metric learning on tabular datasets
'''
from pathlib import Path
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#python_metric_learning imports
# import pytorch_metric_learning as pml
from pytorch_metric_learning import distances, reducers, losses, miners, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

#carla imports
from utils import *


# dataset
class DataFrameDataset(Dataset):
    def __init__(self, x: pd.DataFrame, y: pd.DataFrame):
        # PyTorch
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        self.X_train = torch.tensor(x.to_numpy(), dtype=torch.float32)
        self.Y_train = torch.tensor(y.to_numpy())

    def __len__(self):
        return len(self.Y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.Y_train[idx]

class NumpyDataset(Dataset):
	def __init__(self, x, y):
		super().__init__()
		#create copies
		self.x = torch.tensor(x).to(torch.float32)
		self.y = torch.tensor(y)
	
	def __len__(self):
		return self.y.shape[0]
	
	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]

# model
class Net(nn.Module):
	def __init__(self, input_size, hidden_size=[128,64], output_size=32):
		super().__init__()
		self.layers = nn.Sequential()
		for i, h in enumerate(hidden_size):
			ip = input_size if i==0 else hidden_size[i-1]
			op = hidden_size[i]
			self.layers.add_module(f'layer{i+1}',\
			 nn.Sequential(nn.Linear(ip, op), nn.Tanh(), nn.BatchNorm1d(op)))
		ip = input_size if len(hidden_size)==0 else hidden_size[-1]
		self.layers.add_module(f'embedding_layer', nn.Sequential(nn.Linear(ip, output_size)))

	def forward(self, x):
		return self.layers(x)


#helper functions
def train(model, loss_fn, mining_fn, device, train_loader, optimizer, epoch):
	model.train()
	for batch_idx, (x, y) in enumerate(train_loader):
		x, y = x.to(device), y.to(device)
		embeddings = model(x)
		indices_tuple = mining_fn(embeddings, y)
		loss = loss_fn(embeddings, y, indices_tuple)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if batch_idx%20==0:
			print(f"Epoch: {epoch}, Iteration: {batch_idx}, Loss: {loss},\
			 Number of mined Triplets = {mining_fn.num_triplets}")


def get_all_embeddings(dataset, model):
	tester = testers.BaseTester()
	return tester.get_all_embeddings(dataset, model)

def test(train_set, test_set, model, accuracy_calculator):
	train_embeddings, train_labels = get_all_embeddings(train_set, model)
	test_embeddings, test_labels = get_all_embeddings(test_set, model)
	train_labels = train_labels.squeeze(1)
	test_labels = test_labels.squeeze(1)
	print("Computing Accuracy")
	accuracies = accuracy_calculator.get_accuracy(test_embeddings\
		, test_labels, train_embeddings, train_labels, False)
	print("Test set accuracy (Precision@1)= {}".format(accuracies["precision_at_1"]))


# main function
def train_dml_model(basemodel, args):
	#set seed
	set_random_seed(args.seed)
	dataset_name = args.dataset

	#cuda
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	#hypparams
	_BATCH_SIZE = 256
	_LR = 1e-03
	_NUM_EPOCHS = 10
	_MARGIN = 0.2

	#load data
	target = basemodel.data.target
	#NOTE: feature ordering chosen
	train_X, train_Y = basemodel.get_ordered_features(basemodel.data.df_train), basemodel.data.df_train[target]
	test_X, test_Y = basemodel.get_ordered_features(basemodel.data.df_test), basemodel.data.df_test[target]
	input_size = train_X.shape[1]

	train_set = DataFrameDataset(train_X, train_Y)
	train_loader = DataLoader(train_set, batch_size=_BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
	test_set = DataFrameDataset(test_X, test_Y)
	# testloader = DataLoader(test_set, batch_size=_BATCH_SIZE, shuffle=False)

	model = Net(input_size, [128,128, 128], 64).to(device)

	#if already trained, nothing to do
	if Path(f'../model_cache/{dataset_name}_dml_model.pt').is_file():
		model.load_state_dict(torch.load(f'../model_cache/{dataset_name}_dml_model.pt'))
		return model.to('cpu')

	#else continue and train
	optimizer = optim.AdamW(model.parameters(), lr=_LR)

	distance = distances.CosineSimilarity()
	# distance = distances.LpDistance()
	reducer = reducers.ThresholdReducer(low=0)
	#margin 0.05
	loss_fn = losses.TripletMarginLoss(margin=_MARGIN, triplets_per_anchor="all",\
	 distance=distance, reducer=reducer)
	#other types of triplets are "all" and "hard"
	mining_fn = miners.TripletMarginMiner(margin=_MARGIN, distance=distance, type_of_triplets="hard")
	accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

	for epoch in range(1, _NUM_EPOCHS+1):
		train(model, loss_fn, mining_fn, device, train_loader, optimizer, epoch)
		test(train_set, test_set, model, accuracy_calculator)

	#save model
	torch.save(model.state_dict(), f"../model_cache/{dataset_name}_dml_model.pt")

	return model.cpu()

#finetune dml model
def finetune_dml_model(cf_samples_data, cf_samples_labels, args):
	#set the seed
	set_random_seed(args.seed)
	dataset_name = args.dataset

	#create new model and load pretrained weights
	dml_model = Net(cf_samples_data.shape[1], [128,128,128], 64)
	dml_model.load_state_dict(torch.load(f'../model_cache/{dataset_name}_dml_model.pt'))
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	dml_model = dml_model.to(device)
	dml_model.train()

	#params
	_LR = 1e-03
	_MARGIN = 10
	_NUM_EPOCHS = 1 
	_BATCH_SIZE = 32

	#dataloader
	train_set = NumpyDataset(cf_samples_data, cf_samples_labels)
	# train_loader = DataLoader(train_set, batch_size=cf_samples_data.shape[0], shuffle=True)
	train_loader = DataLoader(train_set, batch_size=_BATCH_SIZE, shuffle=True, drop_last=True)

	#finetune
	optimizer = optim.AdamW(dml_model.parameters(), lr=_LR)
	# distance = distances.CosineSimilarity()
	distance = distances.LpDistance()
	reducer = reducers.ThresholdReducer(low=0)
	loss_fn = losses.TripletMarginLoss(margin=_MARGIN, triplets_per_anchor="all",\
		distance=distance, reducer=reducer)
	mining_fn = miners.TripletMarginMiner(margin=_MARGIN, distance=distance, type_of_triplets="semihard")
	accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

	for epoch in range(1, _NUM_EPOCHS+1):
		train(dml_model, loss_fn, mining_fn, device, train_loader, optimizer, epoch)
		# test(train_set, test_set, model, accuracy_calculator)

	return dml_model.to('cpu')
