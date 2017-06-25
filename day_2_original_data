idxs_finalists = [10,15,19,37,40]
idx_dict_finalists = dict(zip(idxs_finalists, range(len(idxs_finalists))))
ranks_finalists = [[10, 15, 40, 37, 19],
                   [19, 40, 37, 10, 15],
                   [19, 10, 40, 37, 15],
                   [10, 37, 40, 15, 19],
                   [37, 15, 10, 40, 19],
                   [37, 40, 10, 15, 19],
                   [19, 40, 10, 37, 15],
                   [37, 40, 10, 15, 19]
                  ]
ranks_finalists = [[idx_dict_finalists[j] for j in r] for r in ranks_finalists]
print ranks_finalists

n = 5

borda_aggr = borda_reduction(ranks_finalists, n)
borda_score_aggr = borda_reduction_score(ranks_finalists, n)
btl_aggr = btl(ranks_finalists, n)
_, kemeny_aggr, x_lpsolve = kemeny_lpsolve(ranks_finalists, n)
kemeny_brute = rankaggr_brute(ranks_finalists, n)

print 'borda',[idxs_finalists[t] for t in borda_aggr]
#print 'borda using scores', [idxs_finalists[t] for t in borda_score_aggr]
print 'btl', [idxs_finalists[t] for t in btl_aggr]
print 'kemeny', [idxs_finalists[t] for t in kemeny_aggr]
print 'kemeny brute force', [[idxs_finalists[t] for t in ranking] for ranking in kemeny_brute]
