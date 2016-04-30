import numpy as np
import networkx as nx
import random
import math
import matplotlib.pyplot as plt

'''
Construct Jellyfish topology with homogeneity:
All the switches has same number of ports.
All the switches connect to same number of servers.
n : number of switches
k : number of ports per switch
r : number of ports used to connect to other switches
s : seed to generate random number
Reference : "Jellyfish: Networking Data Centers Randomly"
'''
def construct1(n,k,r,s):
	numsw = n
	numsvr = n * (k-r)
	N = numsvr + numsw
	numhash = 10

	G = nx.Graph()

	for i in range(numsvr): # add edges between servers and switches
		svr = i
		sw = numsvr + i/(k-r)
		G.add_edge(svr,sw)

	openPorts = [r for i in range(numsw)] 
	switches_left = numsw 
	consecFails = 0 # If there are ten consecutive fails then we think the constrution period is finished
	random.seed(s)
	while switches_left > 1 and consecFails < 10:
		s1 = random.randrange(numsw)
		while openPorts[s1] == 0:
			s1 = random.randrange(numsw)

		s2 = random.randrange(numsw)
		while openPorts[s2] == 0 or s2==s1:
			s2 = random.randrange(numsw)

		if(G.has_edge(s1+numsvr,s2+numsvr)):
			consecFails += 1
		else:
			consecFails = 0
			G.add_edge(s1+numsvr,s2+numsvr)
			openPorts[s1] -= 1
			openPorts[s2] -= 1
			if(openPorts[s1]==0):
				switches_left -= 1
			if(openPorts[s2]==0):
				switches_left -= 1
	
	if switches_left > 0:
		p1 = numsvr
		p2 = numsvr
		for i in range(numsw):
			if openPorts[i] >= 1:
				p1 = i + numsvr
				openPorts[i] -= 1
				break
		for i in range(numsw):
			if openPorts[i] >= 1:
				p2 = i + numsvr
				openPorts[i] -= 1
				break

		while True:
			rLink = random.choice(list(set(G.edges())-set(range(numsvr))))
			if p1==rLink[0] or p1==rLink[1]:
				continue
			if p2==rLink[0] or p2==rLink[1]:
				continue
			G.remove_edge(rLink[0],rLink[1])
			G.add_edge(p1,rLink[0])
			G.add_edge(p2,rLink[1])
			break

	#draw_graph1(G)
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


	numvlan = 4
	VLAN = [np.zeros((N,N)) for v in range(numvlan)]
	for v in range(numvlan):
		root = random.randrange(numsw) + numsvr
		g = generate_tree(root,G)
		for e in g.edges():
			VLAN[v][e[0]][e[1]] = 1
			VLAN[v][e[1]][e[0]] = 1

	length = numhash * numsvr**2
	stride = numsvr/numhash
	tm = np.zeros((length,1))
	for i in range(length):
		index = i % numsvr**2 # hash class
		hash_id = i / numsvr**2
		src = index/numsvr
		des = index%numsvr
		if(src!=des):
			if src/stride==hash_id:
				tm[i] = 0.01

	shortest_paths = [[] for i in range(numsvr**2)]
	for i in range(numsvr):
		for j in range(numsvr):
			paths_iter = nx.all_shortest_paths(G,i,j)
			paths = []
			for x in paths_iter:
				paths.append(x)
			shortest_paths[i*numsvr+j] = paths

	np.savez("topo.npz",numsvr,numsw,N,numlink,Adj,N2L,C,tm,shortest_paths,numvlan,VLAN,numhash,k,r)



'''
Construct Jellyfish topology with heterogeneity:
Each switch has its own number of port.
Each switch has its own number of port used to connect to other switches.
n : number of switches
k : list of port number of every switch
r : list of number of ports used to connect to other switches
s : seed to generate random number
'''
def construct2(n,k,r,s):
	numsw = n
	numsvr = 0
	for i in range(n):
		numsvr += k-r[i]
	N = numsvr + numsw
	numhash = 8

	G = nx.Graph()

	for i in range(numsvr):# add edges between servers and switches
		svr = i
		sw = numsvr + i/(k-r[i])
		G.add_edge(svr,sw)

	openPorts = r+[]
	switches_left = numsw
	consecFails = 0
	random.seed(s)
	while switches_left > 1 and consecFails < 10:
		s1 = random.randrange(numsw)
		while openPorts[s1] == 0:
			s1 = random.randrange(numsw)

		s2 = random.randrange(numsw)
		while openPorts[s2] == 0 or s2==s1:
			s2 = random.randrange(numsw)

		if(G.has_edge(s1+numsvr,s2+numsvr)):
			consecFails += 1
		else:
			consecFails = 0
			G.add_edge(s1+numsvr,s2+numsvr)
			openPorts[s1] -= 1
			openPorts[s2] -= 1
			if(openPorts[s1]==0):
				switches_left -= 1
			if(openPorts[s2]==0):
				switches_left -= 1
	
	if switches_left > 0:
		p1 = numsvr
		p2 = numsvr
		for i in range(numsw):
			if openPorts[i] >= 1:
				p1 = i + numsvr
				openPorts[i] -= 1
				break
		for i in range(numsw):
			if openPorts[i] >= 1:
				p2 = i + numsvr
				openPorts[i] -= 1
				break

		while True:
			rLink = random.choice(list(set(G.edges())-set(range(numsvr))))
			if p1==rLink[0] or p1==rLink[1]:
				continue
			if p2==rLink[0] or p2==rLink[1]:
				continue
			G.remove_edge(rLink[0],rLink[1])
			G.add_edge(p1,rLink[0])
			G.add_edge(p2,rLink[1])
			break

	#draw_graph(G)
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


	numvlan = 4
	root = [32,33,34,35]
	VLAN = [np.zeros((N,N)) for v in range(numvlan)]
	for v in range(numvlan):
		g = generate_tree(root[v],G)
		#draw_graph(g,root[v])
		for e in g.edges():
			VLAN[v][e[0]][e[1]] = 1
			VLAN[v][e[1]][e[0]] = 1

	length = numhash * numsvr**2
	stride = numsvr/numhash
	tm = np.zeros((length,1))
	for i in range(length):
		index = i % numsvr**2 
		hash_id = i / numsvr**2
		src = index/numsvr
		des = index%numsvr
		if(src!=des):
			if src/stride==hash_id:
				tm[i] = 0.01

	shortest_paths = [[] for i in range(numsvr**2)]
	for i in range(numsvr):
		for j in range(numsvr):
			paths_iter = nx.all_shortest_paths(G,i,j)
			paths = []
			for x in paths_iter:
				paths.append(x)
			shortest_paths[i*numsvr+j] = paths

	np.savez("topo.npz",numsvr,numsw,N,numlink,Adj,N2L,C,tm,shortest_paths,numvlan,VLAN,numhash,k,r)

def generate_tree(root,G):
	nodes_now = [root]
	visited = {}
	for n in G.nodes():
		visited[n] = False
	visited[root] = True

	g = nx.Graph()
	while len(nodes_now)!=0:
		n1 = nodes_now[0]
		del nodes_now[0]
		for n2 in G.neighbors(n1):
			if(not visited[n2]):
				nodes_now.append(n2)
				g.add_edge(n1,n2)
				visited[n2] = True
	return g

def draw_graph1(G):
	deg = 360/20
	pos = {}
	for i in range(20):
		pos[i] = (10 * math.sin(math.radians(i*deg)),10 * math.cos(math.radians(i*deg)))
		pos[i+20] = (7 * math.sin(math.radians(i*deg)),7 * math.cos(math.radians(i*deg)))
	color = []
	size = []
	for i in range(20):
		color.append('yellow')
		size.append(200)
	for i in range(20):
		color.append('red')
		size.append(300)
	nx.draw_networkx(G,pos,node_size=size,node_color=color)
	plt.show()

def draw_graph2(G,rot=None):
	deg = 360/20
	degree_svr = [i*deg for i in range(16)]
	degree_sw = [i*deg for i in range(20)]
	pos = {}
	for i in range(16):
		pos[i] = (10 * math.sin(math.radians(degree_svr[i])),10 * math.cos(math.radians(degree_svr[i])))
	for i in range(20):
		pos[i+16] = (7 * math.sin(math.radians(degree_sw[i])),7 * math.cos(math.radians(degree_sw[i])))
	node_size = []
	node_color = []
	for i in range(16):
		node_size.append(400)
		node_color.append('green')
	for i in range(16):
		node_size.append(500)
		node_color.append('red')
	for i in range(4):
		node_size.append(500)
		node_color.append('blue')

	if(rot):
		node_color[rot] = 'yellow'

	nx.draw_networkx(G,pos,node_size=node_size,node_color=node_color)
	plt.show()

if __name__ == '__main__':
	construct1(20,4,3,0)

	# n = 20
	# k = 4
	# r = [3 for i in range(16)]
	# for i in range(4):
	# 	r.append(4)
	# s = 0
	# construct2(n,k,r,s)