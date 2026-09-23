import numpy as np, json, time
from sim import ba_edges
t0 = time.time(); N = 20000; m = 3; runs = 300
times = np.unique(np.round(np.logspace(np.log10(20), np.log10(N), 30)).astype(int)).tolist()
tr = [10, 100, 1000]
acc = {t: np.zeros((runs, len(times))) for t in tr}
for r in range(runs):
    _, _, _, rec = ba_edges(N, m, seed=1000 + r, track=tr, times=times)
    for t in tr:
        acc[t][r, :] = [k for _, k in rec[t]]
out = {str(t): dict(t=times, mean=acc[t].mean(0).tolist(), std=acc[t].std(0).tolist()) for t in tr}
json.dump(out, open("crescimento.json", "w"))
print("ok", round(time.time() - t0, 1), "s")
