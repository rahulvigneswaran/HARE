'''
Naive Sphere Packing in n-dim
not at all mathematically sound
'''

import numpy as np
import pandas as pd
import time
from scipy.spatial.distance import pdist

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
hyperspherical sampling
'''
def enforce_constraints(P, categorical, immutable):
	_categorical = categorical if categorical is not None else []
	_immutable = immutable if immutable is not None else []
	P[:, _categorical] = 0.
	P[:, _immutable] = 0.
	return P

class Prototypes(nn.Module):
	def __init__(self, N, D):
		#N -> number of prototypes
		#D -> dimension of prototypes
		super().__init__()
		P_init = torch.randn(N,D)
		self.P = nn.Parameter(P_init)
		self.I = nn.Parameter(torch.eye(N), requires_grad=False)

	def forward(self):
		P = F.normalize(self.P, p=2, dim=1)
		return P #return prototypes

class CustomLoss(nn.Module):
	def __init__(self, I, ce=True, base_model=None, cf=None, lamb=0.1):
		super().__init__()
		self.I = I
		self.base_model = base_model
		self.cf = cf  #cf is a torch tensor with ordered features
		self.ce = ce
		self.lamb = lamb

	def forward(self, P):
		M = P @ P.t() - (2 * self.I)
		sim = M.max(dim=1)[0]
		l1 = sim.mean()
		self.mean_sim = l1.detach().clone()

		if self.base_model is not None and self.cf is not None:
			ip = self.cf + P
			probs = self.base_model.predict(ip) #probability that prototype belongs to class 1 
			if self.ce:
				l2 = F.binary_cross_entropy(probs, torch.ones(probs.shape).cuda())
			else:
				# l2 = F.l1_loss(probs, torch.tensor(0.5).cuda())
				l2 = F.hinge_embedding_loss(probs, torch.tensor(-1).cuda(), margin=0.5)

			return l2 + self.lamb * l1

		return l1

#BIG TODO: need to clamp between [0,1] after sampling!!! VV IMP	
def hyperspherical_sampling(cf, base_model, N, D, lamb, continuous=None, categorical=None, immutable=None, epochs=100, lr=1e-01):
	'''
	cf : cf around which to sample
	base_model : generate valid counterfactuals
	N : number of samples to generate
	D : dimensionality of samples
	lamb : tradeoff parameter
	immutables: list of immutable features
	'''
	device = torch.device('cuda:0')
	if isinstance(cf, np.ndarray):
		cf_ = torch.tensor(cf) #make a copy
		cf_ = cf_.to(device)
	else:
		cf_ = torch.tensor(cf.to_numpy()).to(dtype=torch.float32) #make a copy
		cf_ = cf_.to(device)
	cf_ = cf_.view(1,-1)

	prototypes_model = Prototypes(N, D)
	prototypes_model.to(device)

	optimizer = torch.optim.Adam(prototypes_model.parameters(), lr=lr)
	loss_fn = CustomLoss(prototypes_model.I, True, base_model, cf_, lamb)

	for e in range(epochs):
		optimizer.zero_grad()
		l = loss_fn(prototypes_model())
		l.backward()
		optimizer.step()
		if e%100==0:
			print(f'Epoch: {e}, Loss: {l.detach().cpu().item()}, Mean max similarity: {loss_fn.mean_sim.cpu().item()}')

	P = prototypes_model().detach()
	#enforce_constrains
	P = enforce_constraints(P, categorical, immutable)
	sampled_cfs = cf_ + P #final sampled cfs

	#check if they are far away
	print("Mean similarity:", loss_fn.mean_sim.cpu().numpy())
	mask = (base_model.predict(sampled_cfs) >= 0.5).squeeze(1)
	print(f"Number of good counterfactuals sampled is: {mask.sum()} (fraction {mask.sum()/N})")

	#return either np array or pd series
	if isinstance(cf, np.ndarray):
		return sampled_cfs[mask].cpu().numpy()
	else:
		return pd.DataFrame(sampled_cfs[mask].cpu().numpy(), columns=cf.columns)


'''
random sampling on a sphere
'''
def check_overlap(points, r):
	dists = pdist(points, metric='euclidean')
	#check if any distance is within 2*r
	if (dists < 2*r).any():
		return True
	return False

#return centers of spheres
def sphere_packing_centers(d, r, k, immutables, p=0.64):
	'''
	d = dimensionality of sphere
	r = radius of big sphere
	k = number of small spheres to pack
	p = fraction of volume covered by k small spheres
	'''
	_r_optimal = r/np.power(k/p, 1/d)

	init_points = 2*np.random.rand(k,d)-1
	init_points[:, immutables] = 0.
	init_points = r*(init_points/np.linalg.norm(init_points, axis=1, ord=2, keepdims=True))
	_r_delta = 1e-02 #always enough i think

	points = init_points.copy()
	#radius of hard spheres to pack
	_r = 0.0
	stime = time.time()
	while _r < _r_optimal:
		if time.time() - stime > 30:
			break
		#keep increasing radius till there is overlap
		while not check_overlap(points, _r):
			_r += _r_delta
		#now perturb points till there is no overlap
		j=1
		perturb = 1e-02 #always enough i think
		while check_overlap(points, _r):
			if perturb > _r_optimal:
				break
			if j%100==0:
				perturb += 5e-03
			z = 2*np.random.rand(k,d)-1
			z[:, immutables] = 0.
			z = perturb *(z/np.linalg.norm(z, axis=1, ord=2, keepdims=True))
			points = points + z
	#         mask = (np.linalg.norm(points, axis=1, ord=2) > r)
	#         if mask.any():
	#             #some points outside the sphere
	#             points[mask] = r*(points[mask]/np.linalg.norm(points[mask], axis=1, ord=2, keepdims=True))
			j+=1
	
	#push points into radius
	points = r*(points/np.linalg.norm(points, axis=1, ord=2, keepdims=True))

	return points


if __name__=='__main__':
	print(sphere_packing_centers(10,1,5,[0,3,5]))