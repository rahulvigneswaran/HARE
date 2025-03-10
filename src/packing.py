'''
Naive Sphere Packing in n-dim
not at all mathematically sound
'''

import numpy as np
import time
from scipy.spatial.distance import pdist

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