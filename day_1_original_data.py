idxs = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        16, 18, 19, 20, 23, 25, 27, 29, 30, 31, 32, 33,
        34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 49, 50]
idx_dict = dict(zip(idxs, range(len(idxs))))

ranks = [[1, 15], [24, 16, 3, 18], [28, 26], [16, 4, 11, 18], [16, 11, 24, 25, 18],
         [14, 29, 0], [20, 24, 8], [18, 28, 21, 13, 8], [6, 39, 2], [30, 0],
         [6, 13, 5, 22, 31, 34], [6, 5, 27, 18, 34],
         [33, 12, 2, 3], [28, 26, 17],
         [9, 31], [0, 19, 14, 31, 23, 39, 35, 13, 33, 37],
         [22, 25],
         [38, 37, 33, 16, 4, 39, 36, 32, 22, 23, 10, 3, 12, 6],
         [8, 31, 1, 11],  [8, 15, 33, 37], [25, 30, 4, 17], [25, 24, 8],
         [22, 19, 16], [16, 8, 39, 11, 10, 38, 34, 26], [8, 30, 34, 31, 26, 12, 13],
         [18, 6, 31, 36, 5, 12, 16], [1, 10, 30, 0, 20], [26, 16, 34], [28, 13, 31],
         [13, 38, 4, 28, 8, 15], [37, 38, 16, 13, 30, 25, 11, 19, 8, 6], [17, 28, 26],
         [13, 0, 2, 19, 31, 35, 10, 20, 23, 33, 14],
         [31, 0, 19, 23, 35, 13, 14, 2], [36, 32, 38, 13, 37, 17, 4, 8, 5], [16, 28],
         [13, 34, 14, 16, 11, 18], [12, 13, 34, 23, 9], [36, 38, 37, 32, 13, 5, 26, 19]]
'''
dictionary to get from real data to randomized data
dictionary={0: 9,
 1: 26,
 2: 4,
 3: 14,
 4: 5,
 5: 24,
 6: 30,
 7: 31,
 8: 21,
 9: 33,
 10: 23,
 11: 7,
 12: 1,
 13: 39,
 14: 20,
 15: 2,
 16: 29,
 17: 38,
 18: 16,
 19: 37,
 20: 18,
 21: 6,
 22: 8,
 23: 3,
 24: 11,
 25: 10,
 26: 25,
 27: 34,
 28: 32,
 29: 22,
 30: 28,
 31: 15,
 32: 19,
 33: 35,
 34: 13,
 35: 12,
 36: 27,
 37: 36,
 38: 0,
 39: 17}

 '''
print 'How Many Ballots Received:', len(ranks)

length = 0
for r in ranks:
    length += len(r)

print 'Average Ballot Length:', length/len(ranks)

print 'Dictionary of item to number of times it was ranked'
rank_dict = defaultdict(int)
for r in ranks:
    for i in r:
        rank_dict[i] +=1
print sorted([(idxs[x],rank_dict[x]) for x in rank_dict.keys()], key=lambda x:x[1])

print 'Histogram of number of ranks an item appeared in'
freqs = rank_dict.values()
plt.hist(freqs, bins='auto')
plt.title('histogram of number of ranks an item appeared in')
plt.show()

print 'Which indices appear in a ranking.'
idxs_appearing = []
for r in ranks:
    idxs_appearing += r
print len(set(idxs_appearing))


print 'Number of pairwise comparisons an item appeared in'
N = build_N(ranks, 40)
pairwise_counts = np.sum(N, axis=0)+np.sum(N, axis=1).T
print 'pairwise_counts', pairwise_counts

def num_k_m(ranks, k, m):
    counts = []
    ranks = [set(a) for a in ranks]
    for a in ranks:
        c = 0
        for b in ranks:
            if a!= b:
                if len(a.intersection(b)) >= k:
                    c += 1
        if c >= m:
            counts.append(1)
        else:
            counts.append(0)
    return sum(counts)

heatmap = np.zeros((19,20))
for k in range(1,20):
    for m in range(20):
        heatmap[k-1,m] = num_k_m(ranks, k, m)
        if k==2 and m==5:
            print 'for item = 2 and judge =5', heatmap[k,m]

import matplotlib.cm
fig, ax = plt.subplots()
plt.imshow(heatmap, interpolation='none', origin='lower', cmap=matplotlib.cm.coolwarm)
plt.title('Heatmap of the Number of Times That a Judge Ranked \n at least k Items in Common with at least m Other Judges', fontsize=20)
#plt.title('Heatmap of the Number of Times That a Judge Intersected At Least m Other Judges in at Least k Other Items')
plt.ylim(1,20)
plt.colorbar()
plt.xlabel('judges', fontsize=20)
plt.ylabel('items', fontsize=20)
plt.show()


n = 40
borda_aggr = borda_reduction([r[:min(3, len(r))] for r in ranks], n)
borda_score_aggr = borda_reduction_score(ranks, n)



#btl_aggr = btl(ranks, n)
#_, kemeny_aggr, x_lpsolve = kemeny_lpsolve(ranks, n)
#
print 'borda',[idxs[t] for t in borda_aggr]
print 'borda using scores', [idxs[t[0]] for t in borda_score_aggr]
#print 'btl', [t for t in btl_aggr]
#print 'kemeny', [idxs[t] for t in kemeny_aggr]
