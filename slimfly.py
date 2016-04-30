import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import random

'''
Construct Slim Fly topology.
q : the only parameter to construct Slim Fly, must be a prime power.
This file just implement the condition when delta is 1.
Reference : "Slim Fly: A Cost Effective Low-Diameter Network Topology"
'''
def construct(q):
	delta = 1
	if (q-delta)%4!=0:
		raise Exception("Sorry but I can just hadle when delta is 1.")
	w = 0
	while True:
		if((4*w+delta)==q): break
		w = w + 1
	k_ = (3*q-delta)/2
	p = int(math.ceil(k_/2.0))
	k = k_ + p
	numsw = 2*q**2
	numsvr = p*numsw
	N = numsw + numsvr

	epsilon = 1 
	F = set(range(q))
	# find the primitive element epsilon of F
	for e in F:
		if(e==0 or e==1): continue
		if(isPrimitive(e,F)):
			epsilon = e
			break

	X = set()
	X_ = set()
	for i in range(q-1):
		if(i%2==0):
			X.add((epsilon**i)%q)
		else:
			X_.add((epsilon**i)%q)

	G = nx.Graph()
	for i1 in range(2):
		for i2 in range(q):
			for i3 in range(q):
				G.add_node(toNum((i1,i2,i3))+numsvr)

	for i1 in G.nodes():
		for i2 in G.nodes():
			if(i1==i2): continue
			n1 = toAddr(i1-numsvr)
			n2 = toAddr(i2-numsvr)
			if(n1[0]==0 and n2[0]==0 and n1[1]==n2[1] and (n1[2]-n2[2])%q in X):
				G.add_edge(i1,i2)
			if(n1[0]==1 and n2[0]==1 and n1[1]==n2[1] and (n1[2]-n2[2])%q in X_):
				G.add_edge(i1,i2)
			if(n1[0]==0 and n2[0]==1 and n1[2]==(n2[1]*n1[1]+n2[2])%q):
				G.add_edge(i1,i2)

	# draw_graph(G,q,numsvr)

	switches = G.nodes()
	for sw in switches:
		for i in range(p):
			G.add_edge((sw-numsvr)*p+i,sw)


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

	C = np.ones((2 * numlink, 1))

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

def toNum(sw):
	return sw[0]*25+sw[1]*5+sw[2]

def toAddr(sw):
	return (sw/25,sw%25/5,sw%5)

def isPrimitive(a,A):
	n = len(A)
	F = A-set([0])
	flag = True
	for e in F:
		if(not canPresent(a,e,n)):
			flag = False
			break
	return flag

def canPresent(a,b,c):
	flag = False
	for i in range(20):
		t = a**i%c
		if(t==b):
			flag = True
			break
	return flag

def draw_graph(G,q,numsvr):
	pos = {}
	for n in G.nodes():
		n_addr=toAddr(n-numsvr)
		if(n_addr[0]==0):
			pos[n] = (n_addr[1],q-n_addr[2])
		else:
			pos[n] = (n_addr[1]+10,q-n_addr[2])
	nx.draw_networkx(G,pos,node_color='yellow',with_labels=False)
	plt.show()

if __name__ == '__main__':
	construct(5)