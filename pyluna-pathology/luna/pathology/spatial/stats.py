from scipy.spatial.distance import cdist
import numpy as np

# Helper function to unlist singleton arrays and handle the ls parameter
def _ret(count, ls):
	if ls:
		return count[0] if len(count) == 1 else count
	else:
		m = np.mean(count, axis=1)
		return m[0] if len(m) == 1 else m

# # Compute an edge-correction weight for each phenotype 1
# # cell near the edge of the slide
# DEPRECIATED: < 1-5% effect on total calculation
# def edge_correct(p1XY, R):
# 	# These bounds are the min/max of the (x,y) coordinates for
# 	# phenotype 1 cells, should really be the true edge of the fov
# 	xbound = [np.min(p1XY.T[0]),np.max(p1XY.T[0])]
# 	ybound = [np.min(p1XY.T[1]),np.max(p1XY.T[1])]
# 	weights = []
# 	for cell in p1XY:
# 		areas = np.zeros(2)
# 		# Check the X and Y edges
# 		for i in range(2):
# 			OP = np.min(np.abs(xbound-cell[i]))
# 			if OP < R:
# 				theta = 2 * np.arccos(OP/R)
# 				sector = R**2 * theta / 2
# 				triangle = 1/2 * R**2 * np.sin(theta)
# 				areas[i] = sector - triangle
# 		circlearea = np.pi * R**2
# 		prop = (circlearea - areas)/circlearea
# 		# Will be correct for cells near one edge but only
# 		# an approximation for cells near two edges
# 		weights.append(np.max(1/prop))
# 	return weights

def Kfunction(p1XY, p2XY, radius, ls = False, count=True, intensity=[], distance = False, distance_scale=10.0):
	''' Computes the Counting, Intensity, and experimental
		Intensity-Distance K functions

    Args:
	    p1XY (np.ndarray): An Nx2 array representing the (X,Y) coordinates of cells with phenotype 1
	    p2XY (np.ndarray): Same as p1XY but for phenotype 2 cells
	    radius (float, list[float]): The radius (or list of radii) to consider
	    ls (bool): If True, returns an |radius|x|p1XY| 2D array representing the K function
	    	for each phenotype 1 cell for each radius. If False, returns the mean
	    	for each radius
	    count (bool): By default, this function only computes the Counting K function.
	           Can be disabled with count=False.
	    intensity (np.ndarray): An array of length |p2XY| representing the intensity of each
	               phenotype 2 cell. When passed in, this method will also compute
	               the Intensity K function
        distance (bool): If an intensity array is passed in, then setting distance=True
                  will compute the experimental Intensity-Distance K function
		distance_scale (float): Characteristic distance scale (usually approx. 1 cell length in the given units)

	Returns:
		dict: a dictionary with keys ["count", "intensity", "distance"] and values corresponding to the result of each K function
	'''
	# Compute the distance matrix
	dists = cdist(p1XY,p2XY)

	# Turn radius into an array if it isn't one already
	try:
		it = iter(radius)
	except TypeError:
		radius = [radius]

	# Define the lambdas for each K function variant
	CKfunc  = lambda mask: np.sum(mask, axis=1)
	IKfunc  = lambda Imask: np.sum(Imask, axis=1)
	IDKfunc = lambda Imask: np.sum(Imask*(1/(distance_scale + (dists/distance_scale)**3)), axis=1)

	# Compute the mask for each radius
	masks = [(dists <= r) for r in radius]

	# Calculate each K function
	Kdict = {}
	if count:
		CK = [CKfunc(mask) for mask in masks]
		Kdict["count"] = _ret(CK,ls)
	if len(intensity) > 0:
		assert(len(intensity) == len(p2XY))
		Imasks = [mask*intensity for mask in masks]
		IK = [IKfunc(Imask) for Imask in Imasks]
		Kdict["intensity"] = _ret(IK,ls)
		if distance:
			IDK = [IDKfunc(Imask) for Imask in Imasks]
			Kdict["distance"] = _ret(IDK,ls)

	return Kdict