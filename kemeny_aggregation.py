#from __future__ import print_function
import numpy as np
from itertools import combinations, permutations
from lpsolve55 import lpsolve
from lp_solve import lp_solve

#input: rank_a[i] (and similarly rank_b) is a vector containing the rank of item
#i where 1 indicates the best, 2 the second best, and so on. if item i is unranked, set rank_a[i] = -1
# def kendall_tau_distance(rank_a, rank_b):
#     KT_distance = 0
#     num_items = rank_a.shape[0]
#
#     for i in range(num_items):
#         for j in range(i+1, num_items):
#             #both items have been ranked by both judges
#             if(rank_a[i] !=-1 and rank_j[] and rank_b[j]!=-1):
#
#             #if only one item has been ranked
#             elif(rank_a[i]!=-1 and rank_b[j]==-1):
#
#             #if only one item has been ranked
#             elif(rank_a[i]==-1 and rank_b[j]]!=-1):

'''
input is a list of lists, where each sublist is a ranking i.e. if i have 50 items
something like,  [34, 3, 9, 12]

DONE - From this we will build our N matrix
build objective <- N

'''

def build_N(rankings, n, extend=False):
    N = np.zeros((n,n))
    for r in rankings:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                N[r[i],r[j]] += 1
    return N

def build_objective(N, extend=False):
    return N.ravel()

def build_constraints(n, solver='lpsolve'):
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
        constraint_signs = np.hstack([np.zeros(len(pairwise_constraints)),  # ==
        return constraints, constraint_rhs, constraint_signs
    else solver=='glpk':
        G = -1*triangle_constraints
        h = -1*np.ones(len(G))
        A = pairwise_constraints
        b = np.ones(len(A))
        B = set()
        return G,h,A,b
    else:
        raise

# def kendalltau_dist(rank_a, rank_b):
#     tau = 0
#     n_candidates = len(rank_a)
#     for i, j in combinations(range(n_candidates), 2):
#         tau += (np.sign(rank_a[i] - rank_a[j]) ==
#                 -np.sign(rank_b[i] - rank_b[j]))
#     return tau
#
# def rankaggr_brute(ranks):
#     min_dist = np.inf
#     best_rank = None
#     n_voters, n_candidates = ranks.shape
#     for candidate_rank in permutations(range(n_candidates)):
#         dist = np.sum(kendalltau_dist(candidate_rank, rank) for rank in ranks)
#         if dist < min_dist:
#             min_dist = dist
#             best_rank = candidate_rank
#     return min_dist, best_rank
#
# def _build_graph(ranks):
#     n_voters, n_candidates = ranks.shape
#     edge_weights = np.zeros((n_candidates, n_candidates))
#     for i, j in combinations(range(n_candidates), 2):
#         preference = ranks[:, i] - ranks[:, j]
#         h_ij = np.sum(preference < 0)  # prefers i to j
#         h_ji = np.sum(preference > 0)  # prefers j to i
#         if h_ij > h_ji:
#             edge_weights[i, j] = h_ij - h_ji
#         elif h_ij < h_ji:
#             edge_weights[j, i] = h_ji - h_ij
#     return edge_weights
#
# def rankaggr_lp(ranks):
#     """Kemeny-Young optimal rank aggregation"""
#
#     n_voters, n_candidates = ranks.shape
#
#     # maximize c.T * x
#     edge_weights = _build_graph(ranks)
#     c = -1 * edge_weights.ravel()
#
#     idx = lambda i, j: n_candidates * i + j
#
#                                   np.ones(len(triangle_constraints))])  # >=
#     np.set_printoptions(threshold=np.nan)
#     print c
#     print constraint_signs, n_candidates
#     obj, x, duals = lp_solve(c, constraints, constraint_rhs, constraint_signs.T)
#                              #xint=range(1, 1 + n_candidates ** 2))
#
#     x = np.array(x).reshape((n_candidates, n_candidates))
#     aggr_rank = x.sum(axis=1)
#
#     return obj, aggr_rank
# cols = "Alicia Ginny Gwendolyn Robin Debbie".split()
# ranks = np.array([[0, 1, 2, 3, 4],
#                   [0, 1, 3, 2, 4],
#                   [4, 1, 2, 0, 3],
#                   [4, 1, 0, 2, 3],
#                   [4, 1, 3, 2, 0]])
#
# dist, aggr = rankaggr_brute(ranks)
# print("BRUTE Kemeny-Young aggregation with score {} is: {}".format(
# dist,
# ", ".join(cols[i] for i in np.argsort(aggr))))
#
#
# _, aggr = rankaggr_lp(ranks)
# score = np.sum(kendalltau_dist(aggr, rank) for rank in ranks)
# print("LP Kemeny-Young aggregation with score {} is: {}".format(
#     score,
#     ", ".join(cols[i] for i in np.argsort(aggr))))
#
# print(_build_graph(ranks))
