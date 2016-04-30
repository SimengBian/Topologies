import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

'''
Construct BCube topology.
n : number of servers connect to one switch
k : number of layers(count from 0)
'''
def construct(n,k):
	numlayer = k+1
	numsvr = n**(k+1)
	numsw_p = n**k
	numsw = numlayer * numsw_p
	N = numsvr + numsw
	numflow = 8

	G = nx.Graph()
	for i1 in range(numsvr):
		i2 = i1 + 1
		while(i2<numsvr):
			if(hammingDistance(i1,i2,n,k)==1):
				commonsw = getCommonSwitch(i1,i2,n,k)
				G.add_edge(i1,commonsw)
				G.add_edge(i2,commonsw)
			i2 = i2 + 1

	#draw_graph(G,n,numsvr,numsw_p)

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
	#print numlink
	#print N2L
	C = np.ones((2 * numlink, 1))

	length = numflow * numsvr**2
	tm = np.zeros((length,1))
	for i in range(length):
		index = i % numsvr**2
		i1 = index/numsvr
		i2 = index%numsvr
		if(hammingDistance(i1,i2,n,k)==k+1):
			tm[i] = 0.1

	shortest_paths = [[] for i in range(numsvr**2)]
	for i in range(numsvr):
		for j in range(numsvr):
			paths_iter = nx.all_shortest_paths(G,i,j)
			paths = []
			for x in paths_iter:
				paths.append(x)
			shortest_paths[i*numsvr+j] = paths

	#print shortest_paths[7]
	np.savez("topo.npz",numsvr,numsw,N,numlink,Adj,N2L,C,tm,shortest_paths)

def hammingDistance(x,y,n,k):
	x_addr = getSvrAddr(x,n,k) + []
	y_addr = getSvrAddr(y,n,k) + []
	d = 0
	for i in range(k+1):
		if x_addr[i]!=y_addr[i]:
			d = d + 1
	return d

def getSvrAddr(x,n,k):
	num = x
	mid = [0 for i in range(k+1)]
	i = k
	while i>=0:
		if num==0:
			mid[i] = 0
			break
		num,rem = divmod(num,n)
		mid[i] = rem
		i = i - 1
	return mid + []

def getSwitchAddr(x,n,k):
	temp = x - n**(k+1)
	l,num = divmod(temp,n**k)
	mid = [0 for i in range(k)]
	i = k - 1
	while i>=0:
		if num==0:
			mid[i] = 0
			break
		num,rem = divmod(num,n)
		mid[i] = rem
		i = i - 1
	mid.insert(0,l)
	return mid + []

def switchAddr2ID(addr,n,k):
	a = addr + []
	ID = n**(k+1)
	ID = ID + a[0] * n**k
	i = 1
	while(i<=k):
		ID = ID + a[i]*n**(k-i)
		i = i + 1
	return ID

def getCommonSwitch(x,y,n,k):
	x_addr = getSvrAddr(x,n,k)
	y_addr = getSvrAddr(y,n,k)
	for i in range(k+1):
		if(x_addr[i]!=y_addr[i]):
			l = k-i # common switch layer
	temp = x_addr + []
	del temp[k-l]
	commonsw_addr = []
	commonsw_addr.append(l)
	commonsw_addr = commonsw_addr + temp
	commonsw = switchAddr2ID(commonsw_addr,n,k)
	return commonsw

def draw_graph(G,n,numsvr,numsw_p):
	pos = {}
	for i in range(numsvr):
		pos[i] = (i,0)
	for i in range(numsw_p):
		pos[i+numsvr] = (i*n+n/2,2)
	for i in range(numsw_p):
		pos[i+numsw_p+numsvr] = (i*n+n/2,4)
	#print pos
	nx.draw_networkx(G,pos)
	plt.show()

if __name__ == '__main__':
	construct(4,1)