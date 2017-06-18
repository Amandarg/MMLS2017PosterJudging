#from __future__ import print_function


def build_N(rankings, n, extend=False):
    N = np.zeros((n,n))
    for r in rankings:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                N[r[i],r[j]] += 1
    return N

def build_objective(N, extend=False):
    return N.ravel()

def build_constraints(n, solver):
    # constraints for every pair
    idx = lambda i, j: n * i + j

    #n choose 2 rows by n^2. rows are constraints and columns correspond to variables
    pairwise_constraints = np.zeros(((n * (n - 1)) / 2, n ** 2))

    for row, (i, j) in zip(pairwise_constraints, combinations(range(n), 2)):
        row[[idx(i, j), idx(j, i)]] = 1

    # and for every cycle of length 3
    triangle_constraints = np.zeros(((n * (n - 1) *(n - 2)), n ** 2))
    for row, (i, j, k) in zip(triangle_constraints, permutations(range(n), 3)):
        row[[idx(i, j), idx(j, k), idx(k, i)]] = 1


    if solver=='lpsolve':

        constraints = np.vstack([pairwise_constraints, triangle_constraints])
        constraint_rhs = np.hstack([np.ones(len(pairwise_constraints)),
                                    np.ones(len(triangle_constraints))])
        constraint_signs = np.hstack([np.zeros(len(pairwise_constraints)),
                                      np.ones(len(triangle_constraints))])   # ==
        return constraints, constraint_rhs, constraint_signs
    elif solver=='glpk':
        G = -1*triangle_constraints
        h = -1*np.ones(len(G))
        A = pairwise_constraints
        b = np.ones(len(A))
        #B = set(range(n**2))
        #I = set(range(n**2))
        return G,h,A,b#,B,I
    else:
        raise

def kemeny_lpsolve(rankings, n):
    '''Kemeny rank aggregation'''
    from lp_solve import lp_solve
    N = build_N(rankings, n)
    #lp_solve solves a maximization problem, so need to multiply objective by -1
    c = -1*build_objective(N)
    constraints, constraint_rhs, constraint_signs = build_constraints(n, 'lpsolve')
    obj, x, duals = lp_solve(c, constraints, constraint_rhs, constraint_signs.T)
                             #xint=range(1, 1 + n_candidates ** 2))
    x = np.array(x).reshape((n, n))
    aggr_rank = x.sum(axis=0)
    print 'lpsolve aggr_rank', np.argsort(aggr_rank)[::-1]
    return obj, np.argsort(aggr_rank)[::-1], x

def kemeny_glpksolve(rankings,n):
    from cvxopt import matrix
    from cvxopt.glpk import ilp
    #glpk solves a minimization problem
    N = build_N(rankings, n)
    c = build_objective(N)
    G,h,A,b = build_constraints(n, 'glpk')
    status, x = ilp(c, matrix(G), matrix(h), matrix(A), matrix(b),
                    set(range(n**2)), set(range(n**2)))
    x = np.array(x).reshape((n, n))
    aggr_rank = x.sum(axis=0)
    print 'glpk aggr_rank', np.argsort(aggr_rank)[::-1]
    return None, np.argsort(aggr_rank)[::-1], x

def kendalltau_dist(rank, partial):
    d = 0
    rank_inv = dict(zip(rank, range(len(rank)) ))
    for i in range(len(partial)):
        for j in range(i+1, len(partial)):
            if rank_inv[partial[i]] > rank_inv[partial[j]]:
                d+=1
    return d

def rankaggr_brute(ranks, n):
    min_dist = np.inf
    best_rank = None
    count =0
    for candidate_rank in permutations(range(n)):
        if count % 100000 == 0:
            print 'Candidate ', count
        count +=1
        dist = np.sum(kendalltau_dist(candidate_rank, rank) for rank in ranks)
        if dist < min_dist:
            min_dist = dist
            best_rank = candidate_rank
    print 'min_dist', min_dist
    print 'brute aggr', best_rank
    return min_dist, best_rank

def borda_reduction(rankings, n):
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

def sample(rank, k, t):
    n = len(rank)
    ranks = []
    for i in range(t):
        ranks.append([rank[x] for x in sorted(random.sample(range(n),k))])
    return ranks

n =6
tau = range(n); random.shuffle(tau)
print 'tau', tau
ranks = sample(tau, 3, 100)

_, aggr, x_lpsolve = kemeny_lpsolve(ranks, n)
score = np.sum(kendalltau_dist(aggr, rank) for rank in ranks)

aggr = borda_reduction(ranks, n)
print 'borda aggr', aggr

X,y = BTL.get_X_y(ranks, n)
x = BTL.sklearn_mle(X,y)
print 'BTL', np.argsort(x)[::-1]

_, aggr, x_glpk  = kemeny_glpksolve(ranks,n)
print 'GLPK', aggr
