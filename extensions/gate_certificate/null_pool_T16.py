import sys, time
sys.path.insert(0, 'code')
from gatelib import *
name = sys.argv[1]; N = int(sys.argv[2]); lamp = float(sys.argv[3]); tag = sys.argv[4]
sig = float(sys.argv[5]) if len(sys.argv) > 5 else SIG_MEAS
inertia = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
seed0 = int(sys.argv[7]) if len(sys.argv) > 7 else 100000
c0, th, de, coef = nominal_fit(name)
S = []; t0 = time.time()
for i in range(N):
    load, lg = gen_run(name, c0, th, de, seed0 + i, None, 16.0, lamp, inertia)
    S.append(scores(c0, coef, lg, np.random.default_rng(seed0 + 5000000 + i), sig))
    if i % 200 == 0: print(name, tag, i, round(time.time() - t0, 1), flush=True)
S = np.array(S); np.save(OUT / f'null_{name}_{tag}.npy', S)
print('done', name, tag, S.shape, round(time.time() - t0, 1))
