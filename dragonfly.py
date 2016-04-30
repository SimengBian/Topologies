import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
'''
Construct Dradonfly topology.
p : number of ports used to connect server per switch
h : number of ports used to connect to other group per switch
a : number of switches per group
Reference : "Technology-Driven, Highly-Scalable Dragonfly Topology"
'''
def construct(p,h,a):
	numgroup = a*h + 1
	numsw = numgroup * a
	numsvr = numgroup * a * p
	N = numsvr + numsw

	G = nx.Graph()

	for i in range(numsvr): # add edges between servers and switches
		svr = i
		sw = i/p + numsvr
		G.add_edge(svr,sw)

	for i in range(numsw): # add edges between switches in same group
		sw1 = i + numsvr
		for j in range(numsvr):
			sw2 = j + numsvr
			if(i!=j):
				if(i/a==j/a): # i/a mean group ID
					G.add_edge(sw1,sw2)

	next_port = [0 for i in range(numgroup)]

	for g1 in range(numgroup): # add edges between switches in different group
		g2 = g1 + 1
		while g2<numgroup:
			if g1!=g2:
				sw1 = next_port[g1]/h + g1*a + numsvr
				sw2 = next_port[g2]/h + g2*a + numsvr
				next_port[g1] += 1
				next_port[g2] += 1
				G.add_edge(sw1,sw2)
			g2 = g2 + 1

	draw_graph(G,p,h,a)
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

	C = np.zeros((2 * numlink, 1))
	idx = 0
	for i in range(numsvr):
		C[idx] = 1
		C[idx + 1] = 1
		idx += 2
	for j in range(numsvr, numlink):
		C[idx] = 0.01
		C[idx + 1] = 0.01
		idx += 2

	tm = np.zeros((numsvr**2,1))
	for i in range(numsvr**2):
		index = i % numsvr**2 
		src = index/numsvr
		des = index%numsvr
		if(src!=des):
			tm[i] = 0.01

	shortest_paths = [[] for i in range(numsvr**2)]
	for i in range(numsvr):
		for j in range(numsvr):
			paths_iter = nx.all_shortest_paths(G,i,j)
			paths = []
			for x in paths_iter:
				paths.append(x)
			shortest_paths[i*numsvr+j] = paths		

	np.savez("topo.npz",numsvr,numsw,N,numlink,Adj,N2L,C,tm,shortest_paths)

def draw_graph(G,p,h,a):
	pos = {}
	for i in range(a*p*(a*h+1)):
		pos[i] = (i/(a*p)*a*p + i + 5,1)
	for i in range(a*(a*h+1)):
		pos[i+a*p*(a*h+1)] = (pos[i*p][0],random.uniform(2,3))

	nx.draw_networkx(G,pos,with_labels=False,node_size=20,node_color='r')
	plt.show()


if __name__ == '__main__':
	construct(2,2,4)