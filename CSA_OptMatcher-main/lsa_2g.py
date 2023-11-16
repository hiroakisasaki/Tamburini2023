# Code by Fabio Tamburini, https://github.com/ftamburin

import sys
import time
import numpy as np
import random

from scipy.optimize import linear_sum_assignment


def GenProblem(n, m):
	costM = np.random.rand(n, m)
	groupsA = []
	s = 0
	while (s < n):
		g = random.randint(1,20)
		if (s+g > n):
			g = n - s
		groupsA.append(g)
		s += g
	groupsA = np.array(groupsA)
	groupsB = []
	s = 0
	while (s < m):
		g = random.randint(1,20)
		if (s+g > m):
			g = m - s
		groupsB.append(g)
		s += g
	groupsB = np.array(groupsB)
	return costM, groupsA, groupsB
			

def lsa_2g(costM, groupsA, groupsB):
	ncostM = np.zeros((groupsA.shape[0], groupsB.shape[0]))
	ncMidx = np.zeros((groupsA.shape[0], groupsB.shape[0], 2), dtype=np.int32)
	sA = 0
	for j in range(groupsA.shape[0]):
		sB = 0
		for i in range(groupsB.shape[0]):
			#ncostM[j,i] = np.amin(costM[sA:sA+groupsA[j],sB:sB+groupsB[i]])
			idx = np.argmin(costM[sA:sA+groupsA[j],sB:sB+groupsB[i]])
			iA  = idx//groupsB[i]+sA
			iB  = idx%groupsB[i]+sB
			ncMidx[j,i] = np.array([iA, iB])
			ncostM[j,i] = costM[iA, iB]
			sB += groupsB[i]
		sA += groupsA[j]
	row_ind, col_ind = linear_sum_assignment(ncostM)
	x = ncMidx[row_ind, col_ind]
	row_ind = x[:,0]
	col_ind = x[:,1]
	tcost = np.sum(costM[row_ind, col_ind])
	return row_ind, col_ind, tcost




if __name__ == '__main__':

	costM = np.array([[7,1,3,8,9],[4,2,3,9,7],[1,5,6,7,8],[2,1,9,10,0.5],[10,7,5,1,2]])
	groupsA = np.array([2,2,1])
	groupsB = np.array([3,2])

	costM, groupsA, groupsB = GenProblem(int(sys.argv[1]), int(sys.argv[2]))
	startT = time.perf_counter()
	row_ind, col_ind, tcost = lsa_2g(costM, groupsA, groupsB)
	endT = time.perf_counter()
	print('LSA-2G',tcost)
	print('ELAPSED TIME: ',endT-startT)
