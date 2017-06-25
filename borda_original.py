def borda_reduction(rankings, n):
    '''
    Basic Borda Reduction method for computing the winner.
    Computes the pairwise matrix N, and then normalizes the entries by the number of times i was compared to j.
    '''
    N = build_N(rankings, n)
    M = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i!=j:
                total = N[i,j]+N[j,i]
                if total == 0:
                    M[i,j] = 0
                else:
                    M[i,j] = N[i,j]*1.0 / total
    return np.argsort(np.sum(M, axis=1))[::-1]
