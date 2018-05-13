import numpy as np
from scipy.sparse import csr_matrix

def pageRank(G,n,beta = .85, maxerr = .0001):
    """
    Computes the pagerank for each of the n states
    Parameters
    ----------
    G: matrix representing state transitions
       Gji is a binary value representing a transition from state i to j.
    beta: probability of following a transition. 1-s probability of teleporting
       to another state.
    maxerr: if the sum of pageranks between iterations is bellow this we will
            have converged.
    """
    deg_out_beta = np.array(G.sum(axis=0).T/beta)[:,0] #vector: sum of out degree / beta

    # Compute pagerank r until we converge
    ro = np.zeros(n)
    r = np.ones(n)/n
    E = np.ones(n)*((1-beta)/n)

    i=0
    while np.sum(np.abs(r-ro)) > maxerr:
        print("epoch:%d, err:%f"%(i,np.sum(np.abs(r-ro))))
        ro = r.copy()
        r = G.dot(ro/deg_out_beta) # beta*M*r = G*(r/(sum of out degree / beta))
        r = r + E    # r = beta*M*r + [(1-beta)/n]n
        i=i+1
    print("epoch:%d, err:%f"%(i,np.sum(np.abs(r-ro))))
    # return normalized pagerank
    return r


if __name__=='__main__':
	
    E=5105039
    row=np.zeros(E)
    col=np.zeros(E)
    data=np.ones(E)

    i=0
    idmax=0
    with open("web-Google.txt") as file:
    	for l in file:
    		if '#' not in l:
    			arr=l.split('\t')
    			col[i]=int(arr[0])
    			row[i]=int(arr[1])
    			if int(arr[0])>idmax:
    				idmax=int(arr[0])
    			i=i+1
    N=idmax+1


    G=csr_matrix((data, (row, col)),shape=(N, N),dtype=np.float)	
    A=pageRank(G,N)

    b = np.zeros((N,1),dtype = [('importance', float), ('nodeID', int)])
    b['importance'] = A.reshape((N, 1))
    for i in range(0,N):
        b['nodeID'][i]=i
    b=np.sort(b,axis=0,order=['importance'])
    b=b[::-1]
    print(b['nodeID'][0:100])

    f = open('result.txt', 'w')
    for i in range(0,100):
        f.write("%d: %d\r\n"%(i+1,b['nodeID'][i]))
    f.close()