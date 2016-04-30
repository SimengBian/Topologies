import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

'''
Construct n-dimensional Hypercube topology.
'''
def construct(n):
	numsw = 2**n
	numhost = numsw
	N = numsw + numhost
	numflow = 8

	G = nx.Graph()
	for i in xrange(numhost):
		G.add_edge(i,i+numhost)
		for j in xrange(numsw):
			if(hammingDistance(i,j,n)==1):
				G.add_edge(i+numhost,j+numhost)

	Adj = np.zeros((N,N),dtype=int)
	N2L = np.zeros((N,N),dtype=int)
	idx = 0
	for e in G.edges():
		Adj[e[0]][e[1]] = 1
		Adj[e[1]][e[0]] = 1
		N2L[e[0]][e[1]] = idx
		N2L[e[1]][e[0]] = idx + 1
		idx = idx + 2

	numlink = np.sum(Adj) / 2
	C = 0.1 * np.ones((2 * numlink, 1))
	for i in xrange(numhost*2):
		C[i] = 1

	length = numflow * numhost**2
	tm = np.zeros((length,1))
	for i in xrange(length):
		index = i % numhost**2
		i1 = index/numhost
		i2 = index%numhost
		if(i1!=i2):
			tm[i] = 0.01
	#print sum(tm) * 100
	
	shortest_paths = [[] for i in xrange(numhost**2)]
	for i in xrange(numhost):
		for j in xrange(numhost):
			paths_iter = nx.all_shortest_paths(G,i,j)
			paths = []
			for x in paths_iter:
				paths.append(x)
			shortest_paths[i*numhost+j] = paths

	np.savez("topo.npz",numhost,numsw,N,numlink,Adj,N2L,C,tm,shortest_paths)




def ID2Addr(x,n):
	addr = [0 for i in range(n)]
	num = x
	i = n - 1
	while i>=0:
		if num==0:
			addr[i] = 0
			break
		num,rem = divmod(num,2)
		addr[i] = rem
		i = i - 1
	return addr + []

def Addr2ID(l,n):
	ID = 0
	for i in range(n):
		ID = ID + l[i] * 2**(n-i-1)
	return ID

def hammingDistance(x,y,n):
	x_addr = ID2Addr(x,n)
	y_addr = ID2Addr(y,n)
	d = 0
	for i in range(n):
		if x_addr[i]!=y_addr[i]:
			d = d + 1
	return d

if __name__ == '__main__':
	construct(3)