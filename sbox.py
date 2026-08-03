import os
import matplotlib
matplotlib.use("Agg")

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from itertools import combinations

sns.set_theme(context="paper", style="white", font_scale=1.05)

# CONF
SBOX = np.array([
    99,124,119,123,242,107,111,197,48,1,103,43,254,215,171,118,202,130,201,125,
    250,89,71,240,173,212,162,175,156,164,114,192,183,253,147,38,54,63,247,204,
    52,165,229,241,113,216,49,21,4,199,35,195,24,150,5,154,7,18,128,226,235,39,
    178,117,9,131,44,26,27,110,90,160,82,59,214,179,41,227,47,132,83,209,0,237,
    32,252,177,91,106,203,190,57,74,76,88,207,208,239,170,251,67,77,51,133,69,
    249,2,127,80,60,159,168,81,163,64,143,146,157,56,245,188,182,218,33,16,255,
    243,210,205,12,19,236,95,151,68,23,196,167,126,61,100,93,25,115,96,129,79,
    220,34,42,144,136,70,238,184,20,222,94,11,219,224,50,58,10,73,6,36,92,194,
    211,172,98,145,149,228,121,231,200,55,109,141,213,78,169,108,86,244,234,101,
    122,174,8,186,120,37,46,28,166,180,198,232,221,116,31,75,189,139,138,112,
    62,181,102,72,3,246,14,97,53,87,185,134,193,29,158,225,248,152,17,105,217,
    142,148,155,30,135,233,206,85,40,223,140,161,137,13,191,230,66,104,65,153,
    45,15,176,84,187,22
], dtype=np.uint8)
MIXCOL = [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]]


def gmul(a, b):
    p = 0
    for _ in range(8):
        if b & 1:
            p ^= a
        hi = a & 128
        a = (a << 1) & 255
        if hi:
            a ^= 27
        b >>= 1
    return p

def gf_det(mat):
    n = len(mat)
    if n == 1:
        return mat[0][0]
    
    d = 0
    for i in range(n):
        if mat[0][i] == 0:
            continue
        minor = [row[:i] + row[i + 1:] for row in mat[1:]]
        d ^= gmul(mat[0][i], gf_det(minor))
    return d

def popcount(x):
    return bin(int(x)).count("1")

def compute_ddt():
    ddt = np.zeros((256, 256), dtype=np.int32)
    x = np.arange(256, dtype=np.uint8)
    for a in range(256):
        b = SBOX[x] ^ SBOX[x ^ a]
        for v in b:
            ddt[a, v] += 1
    return ddt

def compute_lat():
    lat = np.zeros((256, 256), dtype=np.int32)
    x = np.arange(256)
    sx = SBOX[x].astype(int)
    par = lambda v: np.array([popcount(t) & 1 for t in v])
    xbit = {a: par(a & x) for a in range(256)}
    sbit = {b: par(b & sx) for b in range(256)}
    for a in range(256):
        xa = xbit[a]
        for b in range(256):
            lat[a, b] = np.sum(xa == sbit[b]) - 128
    return lat

def alg_degree():
    deg = 0

    for bit in range(8):
        f = ((SBOX.astype(int) >> bit) & 1).copy()
        anf = f.copy()

        i = 1
        while i < 256:
            for j in range(256):
                if j & 1:
                    anf[j] ^= anf[j ^ i]
            i <<= 1
        d = max((popcount(m) for m in range(256) if anf[m]), default=0)
        deg = max(deg, d)
    return deg

def mds_check():
    minors = 0
    singular = 0

    for i in range(1, 5):
        for rows in combinations(range(4), i):
            for cols in combinations(range(4), i):
                sub = [[MIXCOL[r][c] for c in cols] for r in rows]
                minors += 1
                if gf_det(sub) == 0:
                    singular += 1

    return minors, singular

def branch_weight1():
    worst = 4

    for pos in range(4):
        for val in range(1, 256):
            col = [0, 0, 0, 0]
            col[pos] = val
            out = [0, 0, 0, 0]
            for i in range(4):
                acc = 0
                for j in range(4):
                    acc ^= gmul(MIXCOL[i][j], col[j])
                out[i] = acc
            worst = min(worst, sum(1 for v in out if v))
    return worst

def main():
    os.makedirs("./sbox", exist_ok=True)

    ddt = compute_ddt()
    lat = compute_lat()
    ddt_max = int(ddt[1:, :].max())

    latc = lat.copy()
    latc[0, 0] = 0
    lat_max = int(np.abs(latc).max())
    deg = alg_degree()
    fixed = int(np.sum(SBOX == np.arange(256)))
    opp = int(np.sum(SBOX == (np.arange(256) ^ 255)))
    minors, singular = mds_check()
    branch = 1 + branch_weight1() if singular == 0 else None

    diff_prob = ddt_max / 256.0
    lin_corr = 2 * lat_max / 256.0
    nonlin = 128 - lat_max

    rep = os.path.join("./sbox", "report.txt")
    with open(rep, "w") as f:
        w = lambda *a: f.write(" ".join(str(x) for x in a) + "\n")

        w("=" * 74)
        w("AES S-box and mixcolumns - static cryptographic properties")
        w("=" * 74)
        w(f"Max differential probability = {ddt_max} / 256 = 2^{np.log2(diff_prob):.3f}")
        w("\nS-box linear")
        w(f"Max |LAT| entry = {lat_max} (AES: 16)")
        w(f"Max linear correlation = 2 * {lat_max} / 256 = 2^{np.log2(lin_corr):.3f}")
        w(f"Nonlinearity = 128 - {lat_max} = {nonlin} (AES: 112)")
        w("\nS-box algebraic")
        w(f"Algebraic degree = {deg} (AES: 7)")
        w(f"Fixed points S(x) = x = {fixed} (AES: 0)")
        w(f"Opposite fixed points S(x) = -x = {opp} (AES: 0)")
        w("\nMixColumns diffusion")
        w(f"Square submatricies checked = {minors}")
        w(f"Singular submatricies = {singular}")
        w(f"MDS = {"Yes" if singular == 0 else "No"}")
        w(f"Branch number = {branch} (MDS 4x4: 5)")
        w(f"Minimum output active bytes (weight-1 in) = {branch_weight1()} (=> 1 + 4 = 5)")
        w("\nWide-trail consquence (why the curves saturate at NR = 4)")
        w(f"2 rounds guarantee >= {branch if branch else 5} active S-boxes")
        w("4 rounds guarantee >= 25 active S-boxes")
        w(f"4-round max differential char. probability <= (2^{np.log2(diff_prob):.0f})^25 = 2^{25 * np.log2(diff_prob):.0f}")
        w(f"4-round max linear correlation <= (2^{np.log2(lin_corr):.0f})^25 = 2^{25 * np.log2(lin_corr):.0f}")
    print(open(rep).read())
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(ddt, cmap="magma", vmax=ddt_max)
    ax.set_title(f"AES S-box DDT (max {ddt_max} = 2$^{{-6}}$ diff. probability)")
    ax.set_xlabel("Output difference b")
    ax.set_ylabel("Input difference a")
    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(os.path.join("./sbox", "ddt_heatmap.png"), dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(np.abs(latc), cmap="viridis", vmax=lat_max)
    ax.set_title(f"AES S-box |LAT| (max {lat_max} -> cor 2$^{{-3}}$, NL {nonlin})")
    ax.set_xlabel("Output mask b")
    ax.set_ylabel("Input mask a")
    fig.colorbar(im, ax=ax, label="|Bias count|")
    fig.tight_layout()
    fig.savefig(os.path.join("./sbox", "lat_heatmap.png"), dpi=200)
    plt.close(fig)

main()