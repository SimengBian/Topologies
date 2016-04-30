import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import random

'''
Construct Space Shuffle topology.
n : number of switches
k : number of ports per switch
L : number of spaces
s : seed to generate random number
Reference : "Space Shuffle: A Scalable, Flexible, and High-Bandwidth Data Center Network"
'''
def construct(n,k,L,s):
	numsw = n
	if(k<=2*L):
		raise Exception("Invalid Parameters!")
	k_sw = 2*L
	k_svr = k-k_sw
	numsvr = n * k_svr
	N = numsw + numsvr

	G=nx.Graph()
	servers = range(numsvr)
	for svr in servers:
		G.add_edge(svr,svr/k_svr+numsvr)

	switches = [numsvr+i for i in range(numsw)]
	coordinates = {}
	random.seed(s)
	for sw in switches:
		coordinates[sw]=[]
		for i in range(L):
			coordinates[sw].append(random.random())

	space = {}
	for i in range(L):
		space[i] = []
		for x in coordinates:
			l_now = len(space[i])
			if(l_now==0):
				space[i].append(x)
			elif(coordinates[x][i]<coordinates[space[i][0]][i]):
				space[i].insert(0,x)
			elif(coordinates[x][i]>coordinates[space[i][l_now-1]][i]):
				space[i].append(x)
			else:
				j = 0
				while(j<l_now-1):
					if(coordinates[x][i]>coordinates[space[i][j]][i] and coordinates[x][i]<coordinates[space[i][j+1]][i]):
						space[i].insert(j+1,x)
						break
					else:
						j = j + 1

	free_port = []
	for sw in switches:
		free_port.append(k_sw)

	for i in range(L):
		for sw in switches:
			idx = space[i].index(sw)
			sw1 = space[i][(idx-1)%numsw]
			sw2 = space[i][(idx+1)%numsw]
			if(not G.has_edge(sw,sw1)):
				G.add_edge(sw,sw1)
				free_port[sw-numsvr] -= 1
				free_port[sw1-numsvr] -= 1
			if(not G.has_edge(sw,sw2)):
				G.add_edge(sw,sw2)
				free_port[sw-numsvr] -= 1
				free_port[sw2-numsvr] -= 1

	while(sum(free_port)>1):
		for i in range(numsw):
			if(free_port[i]>0):
				sw1 = i + numsvr
				free_port[i] -= 1
				break
		for i in range(numsw):
			if(free_port[i]>0 and not G.has_edge(sw1,i+numsvr)):
				sw2 = i + numsvr
				free_port[i] -= 1
				break

		G.add_edge(sw1,sw2)

	#draw_graph(G,numsw,numsvr,k_svr)

	
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

	C = 0.01 * np.ones((2 * numlink, 1))

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
	

def draw_graph(G,numsw,numsvr,k_svr):
	servers = range(numsvr)
	switches = [numsvr+i for i in range(numsw)]
	edge_sw = []
	edge_svr = []
	for e in G.edges():
		if(e[0] in switches and e[1] in switches):
			edge_sw.append(e)
		if(e[0] in servers or e[1] in servers):
			edge_svr.append(e)
	pos_sw = {}
	degree_sw = 360/float(numsw)
	for sw in switches:
		pos_sw[sw]=(5*math.sin(math.radians((sw-numsvr)*degree_sw)),5*math.cos(math.radians((sw-numsvr)*degree_sw)))
	
	pos_svr = {}
	for svr in servers:
		sw = svr/k_svr + numsvr
		i = svr%k_svr
		pos_svr[svr] = (6*math.sin(math.radians((sw-numsvr)*degree_sw+10*i)),6*math.cos(math.radians((sw-numsvr)*degree_sw+10*i)))
	pos_all = dict(pos_svr.items()+pos_sw.items())
	nx.draw_networkx_nodes(G,pos=pos_sw,nodelist=switches,with_labels=True)
	nx.draw_networkx_nodes(G,pos=pos_svr,nodelist=servers,node_size=100,node_color='grey')
	nx.draw_networkx_edges(G,pos=pos_sw,edgelist=edge_sw)
	nx.draw_networkx_edges(G,pos=pos_all,edgelist=edge_svr)
	plt.savefig("topo.png")
	plt.show()

if __name__ == '__main__':
	construct(9,6,2,0)