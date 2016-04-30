import numpy as np
'''
Construct fat-tree topology.
k : the number of ports per switch
'''
def construct(k):
	numhost = k**3 / 4
	numedge = k**2 / 2
	numagg = k**2 / 2
	numcore = k**2 / 4
	numsw =  5*(k**2) / 4
	numhash = 20
	N = numhost + numsw
	#print numhost,numsw,N
	hosts = [x for x in range(0,numhost)]
	edges = [x for x in range(numhost,numhost+numedge)]
	aggs = [x for x in range(numhost+numedge,numhost+numedge+numagg)]
	cores = [x for x in range(numhost+numedge+numagg,N)]
	#print cores
	Adj = np.zeros((N, N), dtype=int)
	N2L = np.zeros((N, N), dtype=int)

	stride = k/2
	s = 0
	e = numhost # edge switch
	idx = 0 # number the links
	for h in hosts: # add link between host and edge
		if(s>=stride):
			s = 0
			e = e + 1
		Adj[h][e] = 1
		Adj[e][h] = 1
		N2L[h][e] = idx
		N2L[e][h] = idx + 1
		s = s + 1
		idx = idx + 2

	for i in range(numedge): # add link between edge ang agg
		edge = edges[i]
		if(i%2==0):
			firstAgg = aggs[i]
			sndAgg = aggs[i+1]
		else:
			firstAgg = aggs[i-1]
			sndAgg = aggs[i]
		Adj[edge][firstAgg] = 1
		Adj[firstAgg][edge] = 1
		Adj[edge][sndAgg] = 1
		Adj[sndAgg][edge] = 1
		N2L[edge][firstAgg] = idx
		N2L[firstAgg][edge] = idx + 1
		N2L[edge][sndAgg] = idx + 2
		N2L[sndAgg][edge] = idx + 3
		idx = idx + 4
	
	start = 0 #the start agg switch core link to
	s = 0
	for i in range(numcore): # add link between agg and core
		if(s >= stride):
			s = 0
			start = start + 1
		j = start
		while(j < numagg):
			Adj[aggs[j]][cores[i]] = 1
			Adj[cores[i]][aggs[j]] = 1
			N2L[aggs[j]][cores[i]] = idx
			N2L[cores[i]][aggs[j]] = idx + 1
			idx = idx + 2
			j = j + stride
		s = s + 1
	
	numlink = np.sum(Adj) / 2 # 48
	C = np.zeros((2 * numlink, 1))
	idx = 0
	for i in range(numhost):
	    C[idx] = 1
	    C[idx + 1] = 1
	    idx += 2
	for j in range(numhost, numlink):
	    C[idx] = 0.01
	    C[idx + 1] = 0.01
	    idx += 2

	def isSameEdge(s,d):
		flag = False
		for i in edges:
			if(Adj[s][i]==1 and Adj[d][i]==1):
				flag = True
		return flag

	# construct VLAN
	numvlan = numcore
	VLAN = [np.zeros((N, N)) for r in range(numvlan)]
	s = 0
	start = 0
	for v in range(numvlan): # construct the vth VLAN
		activeAgg = []
		core = cores[v]
		if(s >= stride):
			s = 0
			start = start + 1
		j = start
		while(j < numagg):
			VLAN[v][core][aggs[j]] = 1
			VLAN[v][aggs[j]][core] = 1
			activeAgg.append(j)
			j = j + stride
		s = s + 1 # use for next VLAN
		
		for i in activeAgg:
			agg = aggs[i]
			if(i%2==0):
				firstEdge = edges[i]
				sndEdge = edges[i+1]
			else:
				firstEdge = edges[i-1]
				sndEdge = edges[i]
			VLAN[v][agg][firstEdge] = 1
			VLAN[v][firstEdge][agg] = 1
			VLAN[v][agg][sndEdge] = 1
			VLAN[v][sndEdge][agg] = 1
		
		st = 0
		e = numhost # edge switch
		for h in hosts: # add link between host and edge
			if(st >= stride):
				st = 0
				e = e + 1
			VLAN[v][h][e] = 1
			VLAN[v][e][h] = 1
			st = st + 1

	length = numhash * numhost**2
	tm = np.zeros((length,1))
	for i in range(length):
		index = i % numhost**2
		i1 = index/numhost
		i2 = index%numhost
		if((i1<numhost/2 and i2<numhost/2) or (i1>=numhost/2 and i2>=numhost/2)):
			if(i1==i2):
				tm[i] = 0
			elif(not isSameEdge(i1,i2)):
				tm[i] = 0.01
	
	np.savez("topo.npz",numhost,numsw,N,numlink,Adj,N2L,C,tm,k)
if __name__ == '__main__':
	construct(4)
	# r = np.load("topo.npz")
	# tm = r["arr_7"]
	# print len(tm)
