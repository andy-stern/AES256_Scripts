import os
import re
import glob
import math
import zlib
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from PIL import Image

sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight", "font.family": "DejaVu Sans", "axes.titleweight": "bold", "axes.edgecolor": "#333333", "axes.linewidth": 0.8, "grid.alpha": 0.25})

# CONF
BLUE = "#2b6cb0"
ORANGE = "#dd6b20"
RED = "#c53030"
GREEN = "#2f855a"
SCALAR_COLS = ["avg_hamming_distance", "hamming_distance"]

def experiments(root):
    found = {}

    for csv in (glob.glob(os.path.join(root, "**", "results.csv"), recursive=True) + glob.glob(os.path.join(root, "**", "trials.csv"), recursive=True)):
        nr_dir = os.path.dirname(csv)
        exp_dir = os.path.dirname(nr_dir)

        m = re.fullmatch(r"(\d+)(?:-\d+)?", os.path.basename(nr_dir))
        if not m:
            continue

        found.setdefault(exp_dir, {"csvs": {}, "dir": exp_dir})
        found[exp_dir]["csvs"][int(m.group(1))] = (nr_dir, os.path.basename(csv))
    return found

def classify(exp):
    nr = min(exp["csvs"])
    nr_dir, csvname = exp["csvs"][nr]

    if csvname == "trials.csv":
        if os.path.exists(os.path.join(nr_dir, "avalanche_counts.bin")):
            return "matrix"
        
        if any(os.path.exists(os.path.join(nr_dir, f)) for f in ("corr_pt_ct.bin", "corr_key_ct.bin", "corr_iv_ct.bin")):
            return "correlation"
        
        return "unknown"

    cols = list(pd.read_csv(os.path.join(nr_dir, csvname), nrows=0).columns)

    if "out_diff_hex" in cols or "out_diff_weight" in cols:
        return "differential"

    if "balanced_bytes" in cols or "fully_balanced" in cols:
        return "integral"

    if any(c.startswith("Hd_bit") for c in cols):
        return "perbit"

    if "avg_hamming_distance" in cols:
        return "scalar"

    if "hamming_distance" in cols and "nrnext" in cols:
        return "nrnext"

    if "hamming_distance" in cols:
        return "scalar"

    return "unknown"

def load_curve(exp):
    frames = []

    for nr, (nr_dir, csvname) in exp["csvs"].items():
        p = os.path.join(nr_dir, csvname)
        head = pd.read_csv(p, nrows=0)

        val = next((c for c in SCALAR_COLS if c in head.columns), None)
        use = [c for c in ("nr", "nrnext") if c in head.columns] + ([val] if val else [])

        df = pd.read_csv(p, usecols=use)
        if "nr" not in df.columns:
            df["nr"] = nr

        frames.append(df.rename(columns={val: "value"})[["nr", "value"]])

    return pd.concat(frames, ignore_index=True)

def pick_nr(exp, want):
    ks = sorted(exp["csvs"])
    return want if (want in exp["csvs"]) else ks[-1]

def numtrials(nr_dir, csvname="trials.csv"):
    with open(os.path.join(nr_dir, csvname)) as f:
        return max(1, sum(1 for _ in f) - 1)

def block_reduce(a, target=768):
    r = max(1, a.shape[0] // target)
    c = max(1, a.shape[1] // target)
    a = a[:(a.shape[0] // r) * r, :(a.shape[1] // c) * c]
    return a.reshape(a.shape[0] // r, r, a.shape[1] // c, c).mean((1, 3))

def sample_hex(csv_path, col, nrows):
    try:
        s = pd.read_csv(csv_path, usecols=[col], nrows=nrows)[col].dropna().astype(str)
    except Exception:
        return b""

    joined = "".join(s.tolist())
    if len(joined) % 2:
        joined = joined[:-1]

    try:
        return bytes.fromhex(joined)
    except Exception:
        return b""

def runs_test(sign):
    sign = sign[sign != 0]
    if len(sign) < 2:
        return np.nan, np.nan

    runs = 1 + int(np.sum(sign[1:] != sign[:-1]))
    n1 = int(np.sum(sign > 0))
    n2 = int(np.sum(sign < 0))
    if not (n1 and n2):
        return np.nan, np.nan

    mu = 2 * n1 * n2 / (n1 + n2) + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    z = (runs - mu) / math.sqrt(var) if var > 0 else np.nan
    return z, 2 * stats.norm.sf(abs(z))

def holm(pvals):
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)

    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)

    adj = np.full_like(p, np.nan)
    prev = 0.0
    for i, j in enumerate(order):
        val = min(1.0, (m - i) * p[j])
        prev = max(prev, val)
        adj[j] = prev
    return adj

def rng_batt(data):
    if len(data) < 16:
        return None

    b = np.frombuffer(data, dtype=np.uint8)
    nb = len(b)
    bits = np.unpackbits(b)
    n = len(bits)
    ones = int(bits.sum())

    counts = np.bincount(b, minlength=256).astype(float)
    p = counts / nb
    nz = p[p > 0]

    H = float(-(nz * np.log2(nz)).sum())
    Hmin = float(-np.log2(p.max()))

    s_obs = abs(ones - (n - ones)) / math.sqrt(n)
    p_monobit = math.erfc(s_obs / math.sqrt(2))
    exp = nb / 256.0

    chi2 = float(((counts - exp) ** 2 / exp).sum())
    p_chi2 = float(stats.chi2.sf(chi2, 255))

    pi = ones / n
    if abs(pi - 0.5) < 2 / math.sqrt(n):
        vn = 1 + int(np.sum(bits[1:] != bits[:-1]))
        p_runs = math.erfc(abs(vn - 2 * n * pi * (1 - pi)) / (2 * math.sqrt(2 * n) * pi * (1 - pi)))
    else:
        p_runs = 0.0

    x = b.astype(float)
    sc = float("nan")

    if nb > 2 and x[:-1].std() > 0 and x[1:].std() > 0:
        sc = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    comp = len(zlib.compress(data, 9)) / nb

    return dict(nbytes=nb, entropy=H, min_entropy=Hmin, ones_frac=ones / n, p_monobit=p_monobit, byte_chi2=chi2, p_chi2=p_chi2, p_runs=p_runs, serial_corr=sc, comp_ratio=comp)

def save_bit_bitmap(data, path, width=256):
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    h = len(bits) // width
    if h == 0:
        return None

    img = (1 - bits[:h * width].reshape(h, width)) * 255
    Image.fromarray(img.astype(np.uint8), "L").save(path)
    return path

def save_byte_bitmap(data, path, width=256):
    b = np.frombuffer(data, dtype=np.uint8)
    h = len(b) // width
    if h == 0:
        return None

    Image.fromarray(b[:h * width].reshape(h, width), "L").save(path)
    return path

class Report:
    def __init__(self, path):
        self.f = open(path, "w")

    def h(self, s):
        self.f.write("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78 + "\n")

    def s(self, s):
        self.f.write("\n" + s + "\n" + "-" * len(s) + "\n")

    def w(self, *a):
        self.f.write(" ".join(str(x) for x in a) + "\n")

    def close(self):
        self.f.close()

def analyze_raw(exp, outdir, strials):
    nr = pick_nr(exp, None)
    nr_dir, csvname = exp["csvs"][nr]
    csv = os.path.join(nr_dir, csvname)

    head = pd.read_csv(csv, nrows=0).columns
    streams = [(c.replace("_hex", "").replace("_normal", ""), c) for c in ("key_hex", "iv_hex", "plaintext_hex", "ciphertext_hex") if c in head]
    if not streams:
        return

    bdir = os.path.join(outdir, "bitmaps")
    os.makedirs(bdir, exist_ok=True)

    rep = Report(os.path.join(outdir, "randomness.txt"))
    rep.h(f"Randomness Analysis (NR={nr}, up to {strials} trials sampled)")
    rep.w("Streams: key, iv, and plaintext are /dev/urandom (tests the RNG)")
    rep.w("ciphertext tests the AES output. Ideal entropy is 8.0 b / byte, p > 0.01, and comp~1.0.\n")

    panel = []

    for label, col in streams:
        data = sample_hex(csv, col, strials)
        r = rng_batt(data)
        if r is None:
            continue

        rep.s(f"{label} ({r["nbytes"]} bytes)")
        rep.w(f"Shannon entropy: {r["entropy"]:.5f} bits / byte (ideal 8.0)")
        rep.w(f"Min-entropy: {r["min_entropy"]:.5f} bits / byte")
        rep.w(f"Monobit ones fraction: {r["ones_frac"]:.6f} p = {r["p_monobit"]:.4g}")
        rep.w(f"Byte chi2 (255dof): {r["byte_chi2"]:.1f} p = {r["p_chi2"]:.4g}")
        rep.w(f"Runs test: p = {r["p_runs"]:.4g}")
        rep.w(f"Serial corr (lag1): {r["serial_corr"]:+.5f}")
        rep.w(f"Zlib compression ratio: {r["comp_ratio"]:.4f} (ideal ~1.0 = incompressible)")

        save_bit_bitmap(data, os.path.join(bdir, f"{label}_bits.png"))
        save_byte_bitmap(data, os.path.join(bdir, f"{label}_bytes.png"))
        panel.append((label, data, r))
    rep.close()

    if panel:
        n = len(panel)
        fig, axes = plt.subplots(2, n, figsize=(3.4 * n, 6.2), squeeze=False)

        for i, (label, data, r) in enumerate(panel):
            bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
            h = min(256, len(bits) // 256)

            axes[0, i].imshow(bits[:h * 256].reshape(h, 256), cmap="binary", interpolation="nearest")
            axes[0, i].set_title(f"{label} bitmap\nH = {r["entropy"]:.3f} b / byte")
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])

            b = np.frombuffer(data, dtype=np.uint8)
            axes[1, i].hist(b, bins=256, color=BLUE, alpha=0.85)
            axes[1, i].axhline(len(b) / 256, ls="--", color=RED, lw=1)
            axes[1, i].set_title(f"byte histogram (chi2 p = {r["p_chi2"]:.2g})")
            axes[1, i].set_xlabel("byte value")
        fig.suptitle(f"{os.path.basename(outdir)} - raw-data randomness", fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "raw_data_bitmaps.png"))
        plt.close(fig)

def analyze_scalar(exp, name, outdir, bits, cfrom):
    df = load_curve(exp)
    g = (df.groupby("nr")["value"].agg(["mean", "std", "count", "min", "max"]).reset_index().sort_values("nr"))
    g["sem"] = g["std"] / np.sqrt(g["count"])
    g["pct"] = 100 * g["mean"] / bits

    nr = g["nr"].to_numpy(float)
    mean = g["mean"].to_numpy()
    std = g["std"].to_numpy()
    sem = np.where(g["sem"] == 0, np.nan, g["sem"]).astype(float)

    cm = nr >= cfrom
    cnr = nr[cm]
    cmean = mean[cm]
    cstd = std[cm]
    csem = sem[cm]
    k = len(cnr)
    w = 1 / csem ** 2
    plateau = float(np.nansum(w * cmean) / np.nansum(w) if k else float("nan"))

    rep = Report(os.path.join(outdir, "report.txt"))
    rep.h(f"Experiment: {name} (scalar avalanche)")
    rep.w(f"NR range {int(nr.min())} to {int(nr.max())} with {len(nr)} values. Measured bits = {bits}, trials / NR = {int(g["count"].min())} to {int(g["count"].max())}")
    rep.s("Per-NR Summary (Hamming distance, bits)")
    rep.w(f"{"NR":>4} {"mean":>10} {"std":>8} {"sem":>8} {"min":>7} {"max":>7} {"%diff":>7}")

    for _, row in g.iterrows():
        rep.w(f"{int(row["nr"]):>4} {row["mean"]:>10.3f} {row["std"]:>8.3f} {row["sem"]:>8.4f} {row["min"]:>7.0f} {row["max"]:>7.0f} {row["pct"]:>7.3f}")

    rep.s("Saturation")
    sat_round = next((int(n) for n, m in zip(nr, mean) if abs(m - plateau) < 1.0), None)
    rep.w(f"Plateau (Weighted, NR >= {cfrom}) = {plateau:.4f} bits ({100 * plateau / bits:.3f}%)")
    rep.w(f"First NR within 1 bit of plateau = {sat_round}")
    rep.w(f"Reference: 50% = {bits / 2:.0f} bits. Plaintext CBC cap ~ 1056 or 25.78%")

    stat_p = {}
    if k >= 3:
        z = (cmean - plateau) / csem
        chi2 = float(np.nansum(z ** 2))
        dof = k - 1
        p_hom = (stats.chi2.sf(chi2, dof))
        between = float(np.nanstd(cmean, ddof=1))
        within = float(np.sqrt(np.nanmean(csem ** 2)))

        sl_m, _, _, p_m, _ = stats.linregress(cnr, cmean)
        sl_s, _, _, p_s, _ = stats.linregress(cnr, cstd)
        z_runs, p_runs = runs_test(np.sign(cmean - plateau))
        cusum = np.nancumsum(z)
        cband = 1.96 * np.sqrt(np.arange(1, k + 1))
        p_per = np.nan
        per_text = "N/A (need >= 8 contiguous)"

        if k >= 8 and np.all(np.diff(cnr) == 1):
            dev = cmean - plateau
            P = (np.abs(np.fft.rfft(dev - dev.mean())) ** 2)[1:]
            P = P / P.mean()
            f = np.fft.rfftfreq(k, 1.0)[1:]
            thr = -np.log(1 - 0.95 ** (1.0 / len(P)))
            ip = int(np.argmax(P))
            p_per = float(np.exp(-P[ip]) * len(P))
            per_txt = (f"Peak period {1 / f[ip]:.1f} round, power {P[ip]:.2f} versus thr {thr:.2f} -> {"SIG" if P[ip] > thr else "ns"}")

        rep.s(f"Converged-region stats (NR >= {cfrom}, k = {k})")
        rep.w(f"Between-NR std / within(SEM) = {between:.4f} / {within:.4f}. Ratio = {between / within:.3f} (~1 => sampling_noise)")
        rep.w(f"Homogeneity chi2 / dof = {chi2:.1f} / {dof} = {chi2 / dof:.3f} p = {p_hom:.4g} -> {"flat" if p_hom > 0.05 else "structure"}")
        rep.w(f"Trend in mean = {sl_m:+.5f} bits / round p = {p_m:.4g}")
        rep.w(f"Tightening (std ~ NR) = {sl_s:+.5f} bits / round p = {p_s:.4g} -> {"tightens" if sl_s < 0 and p_s > 0.05 else "no tightening"}")
        rep.w(f"Runs test = z {z_runs:+.2f} p = {p_runs:.4g}")
        rep.w(f"CUSUM max|.| / band = {np.nanmax(np.abs(cusum)):.2f} / {cband[-1]:.2f} -> {"drift" if np.nanmax(np.abs(cusum)) > cband[-1] else "no drift"}")
        rep.w(f"Periodicity = {per_txt}") 

        stat_p = {"homogeneity": p_hom, "trend_mean": p_m, "tightening": p_s, "runs": p_runs, "periodicity": p_per}
        adj = holm(list(stat_p.values()))
        rep.s("Multiple-comparison correction (Holm across converged tests)")

        for (nm, pv), pa in zip(stat_p.items(), adj):
            rep.w(f"{nm:12s} raw p = {pv:.4g}. Holm-adj = {pa:.4g} {"*" if pa < 0.05 else ""}")
            rep.w(f"Min Holm-adjusted p = {np.nanmin(adj):.4g} -> {"a test survives correction" if np.nanmin(adj) < 0.05 else "nothing significant after correction"}")

        rep.s("Avalanche interpretation")
        for target in (1, 2, 3):
            if target in g["nr"].values:
                rep.w(f"NR = {target}: {float(g.loc[g["nr"] == target, "pct"].iloc[0]):.2f}% diffusion")

        rep.w("Low-round rise reflects progressive diffusio. NR >= ~4 is indistinguishable from the plateau at this precision")
    rep.close()

    scalar_figures(name, outdir, bits, cfrom, g, nr, mean, std, sem, cnr, cmean, csem, plateau, df)

def scalar_figures(name, outdir, bits, cfrom, g, nr, mean, std, sem, cnr, cmean, csem, plateau, df):
    cstd = std[nr >= cfrom]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].errorbar(nr, mean, yerr=sem, marker="o", ms=3, lw=1.2, color=BLUE, capsize=2)
    ax[0].axhline(bits / 2, ls="--", color="gray", lw=1)
    ax[0].set_title("Full range")
    ax[0].set_xlabel("NR")
    ax[0].set_ylabel("Mean Hamming dist. (bits)")

    if len(cnr):
        pad = max(6 * np.nanmax(csem), 0.5)
        pad2 = max(2.5 * np.nanmax(csem), 0.15)

        ax[1].errorbar(cnr, cmean, yerr=csem, marker="o", ms=3, lw=1.1, color=BLUE, capsize=2)
        ax[1].axhline(plateau, ls=":", color=ORANGE, lw=1.2)
        ax[1].set_ylim(plateau - pad, plateau + pad)
        ax[1].set_title(f"Medium zoom (NR >= {cfrom})")

        ax[2].errorbar(cnr, cmean, yerr=csem, marker="o", ms=4, lw=1.1, color=BLUE, capsize=3)
        ax[2].axhline(plateau, ls=":", color=ORANGE, lw=1.2, label=f"Plateau {plateau:.3f}")
        ax[2].set_ylim(plateau - pad2, plateau + pad2)
        ax[2].set_title("Tight zoom (+/- few SEM)")
        ax[2].legend(fontsize=8)

    for a in ax:
        a.set_xlabel("NR")

    fig.suptitle(f"{name} - avalanche versus rounds at three scales", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "curve_3scale.png"))
    plt.close(fig)

    k = len(cnr)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(nr, std, marker="o", ms=3, color=BLUE, label="Per-trial STD")
    ax[0, 0].plot(nr, sem, marker="s", ms=3, color=ORANGE, label="SEM")

    if k >= 3:
        sl_s, ic, _, _, _ = stats.linregress(cnr, cstd)
        ax[0, 0].plot(cnr, sl_s * cnr + ic, "--", color=RED, lw=1, label=f"STD trend {sl_s:+.4f} / round")

    ax[0, 0].set_title("Spread versus NR (tightening test)")
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_xlabel("NR")
    ax[0, 0].set_ylabel("Bits")

    if k >= 3:
        z = (cmean - plateau) / csem
        cusum = np.nancumsum(z)
        cb = 1.96 * np.sqrt(np.arange(1, k + 1))

        ax[0, 1].axhspan(-2, 2, color=GREEN, alpha=0.08)
        ax[0, 1].axhspan(-3, 3, color=GREEN, alpha=0.05)
        ax[0, 1].stem(cnr, z)
        ax[0, 1].axhline(0, color=ORANGE)
        ax[0, 1].set_title("Per-NR z versus plateau")
        ax[0, 1].set_xlabel("NR")
        ax[0, 1].set_ylabel("SEM units")

        ax[1, 0].plot(cnr, cusum, marker="o", ms=3, color=BLUE)
        ax[1, 0].plot(cnr, cb, "--", color=RED)
        ax[1, 0].plot(cnr, -cb, "--", color=RED)
        ax[1, 0].axhline(0, color="gray")
        ax[1, 0].set_title("CUSUM (drift)")
        ax[1, 0].set_xlabel("NR")
        ax[1, 0].set_ylabel("Cumulative z")

        if k >= 8 and np.all(np.diff(cnr) == 1):
            dev = cmean - plateau
            P = (np.abs(np.fft.rfft(dev - dev.mean())) ** 2)[1:]
            P = P / P.mean()
            f = np.fft.rfftfreq(k, 1.0)[1:]

            ax[1, 1].stem(1 / f, P)
            ax[1, 1].axhline(-np.log(1 - 0.95 ** (1.0 / len(P))), ls="--", color=RED)
            ax[1, 1].set_title("Periodogram")
            ax[1, 1].set_xlabel("Period (rounds)")
        else:
            ax[1, 1].axis("Off")

    fig.suptitle(f"{name} - convergence diagnostics", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "convergence.png"))
    plt.close(fig)

    pick = [n for n in (1, 2, max(cfrom, 4), int(nr.max())) if n in set(nr.astype(int))]
    fig, axes = plt.subplots(1, len(pick), figsize=(4 * len(pick), 3.6), squeeze=False)

    for ax, n in zip(axes[0], pick):
        vals = df[df["nr"] == n]["value"].to_numpy()
        sns.histplot(vals, bins=40, stat="density", color=BLUE, ax=ax, alpha=0.7)
        mu = vals.mean()
        sd = vals.std()

        xs = np.linspace(vals.min(), vals.max(), 200)
        ax.plot(xs, stats.norm.pdf(xs, mu, sd), color=RED, lw=1.2)
        ax.set_title(f"NR={n} mean={mu:.1f}")
        ax.set_xlabel("Per-trial Hamming dist.")

    fig.suptitle(f"{name} - per-trial Hamming dis. distributions", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "hd_distributions.png"))
    plt.close(fig)

def analyze_matrix(exp, name, outdir, nr_want):
    nr = pick_nr(exp, nr_want)
    nr_dir, _ = exp["csvs"][nr]
    N = numtrials(nr_dir)
    a = np.fromfile(os.path.join(nr_dir, "avalanche_counts.bin"), dtype=np.uint16)
    side = int(math.isqrt(a.size))
    prob = a.reshape(side, side).astype(np.float64) / N

    ii, jj = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    reachable = (jj// 128) >= (ii // 128)
    rvals = prob[reachable]
    uvals = prob[~reachable]
    sac = np.abs(rvals - 0.5)
    se = math.sqrt(0.25 / N)
    chi2 = float((((rvals - 0.5) / se) ** 2).sum())
    dof = reachable.sum()

    rep = Report(os.path.join(outdir, "report.txt"))
    rep.h(f"Experiment: {name} (avalanche dependency matrix)")
    rep.w(f"NR = {nr} matrix {side}x{side} numtrials = {N} per-cell SE = {se:.5f}")
    rep.s("Overall")
    rep.w(f"Grand mean P(flip) = {prob.mean():.5f}")
    rep.w(f"Reachable-region mean P = {rvals.mean():.5f} (ideal 0.5) std = {rvals.std():.5f}")
    rep.w(f"Unreachable-region mean P = {uvals.mean():.6f} (ideal 0.0) max = {uvals.max():.5f}")
    rep.s("Strict avalanche criterion (reachable cells)")
    rep.w(f"Mean |P - 0.5| = {sac.mean():.5f}")
    rep.w(f"Max |P - 0.5| = {sac.max():.5f}")
    rep.w(f"Chi2 versis 0.5 = {chi2:.1f} / dof {dof} reduced {chi2 / dof:.4f}. P = {stats.chi2.sf(chi2, dof):.4g}")

    frac = float(np.mean(sac > 3 * se))
    rep.w(f"Reachable cells with |P - 0.5| > 3SE = {100 * frac:.3f}% (expected ~0.27%)")
    rep.s("Marginals")
    rep.w(f"Per-input-bit P: min = {prob.mean(1).min():.4f}. Mean = {prob.mean(1).mean():.4f}. Max = {prob.mean(1).max():.4f}")
    rep.w(f"Per-output-bit mean P: min = {prob.mean(0).min():.4f}. Mean = {prob.mean(0).mean():.4f}. Max = {prob.mean(0).max():.4f}")
    rep.close()

    disp = block_reduce(prob, 768)
    fig = plt.figure(figsize=(12, 5.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1])
    axh = fig.add_subplot(gs[0, 0])
    im = axh.imshow(disp, origin="lower", cmap="magma", vmin=0, vmax=0, aspect="auto")
    axh.set_title(f"P(output flips | input flipped), NR = {nr}")
    axh.set_xlabel("Output bit")
    axh.set_ylabel("Input bit")
    fig.colorbar(im, ax=axh, fraction=0.046)
    fig.add_subplot(gs[0, 1]).plot(prob.mean(1), np.arange(side), lw=0.7, color=BLUE)
    fig.axes[-1].set_title("Per-input mean")
    fig.axes[-1].axvline(0.5, ls=":", color="gray")
    fig.add_subplot(gs[1, 0]).plot(np.arange(side), prob.mean(0), lw=0.7, color=ORANGE)
    fig.axes[-1].set_title("Per-output mean")
    fig.axes[-1].axhline(0.5, ls=":", color="gray")
    fig.suptitle(f"{name} - avalanche matrix (mean P = {prob.mean():.3f})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "matrix_heatmap.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.4))
    sns.histplot((rvals - 0.5) / se, bins=120, stat="density", color=BLUE, ax=ax, alpha=0.75)
    xs = np.linspace(-5, 5, 200)
    ax.plot(xs, stats.norm.pdf(xs), color=RED, lw=1.3, label="N(0, 1)")
    ax.set_title(f"{name} - reachable-cell (P - 0.5) / SE versus sampling null")
    ax.set_xlabel("Standardized deviation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "matrix_sac_hist.png"))
    plt.close(fig)

def analyze_correlation(exp, name, outdir, nr_want):
    nr = pick_nr(exp, nr_want)
    nr_dir, _ = exp["csvs"][nr]
    N = numtrials(nr_dir)
    floor = 1 / math.sqrt(N)
    pt = os.path.join(nr_dir, "corr_pt_ct.bin")
    cols = int(math.isqrt(np.fromfile(pt, dtype=np.float32).size))
    mats = []

    for fn, rh in (("corr_pt_ct.bin", None), ("corr_key_ct.bin", 256), ("corr_iv_ct.bin", 128)):
        p = os.path.join(nr_dir, fn)
        arr = np.fromfile(p, dtype=np.float32)
        c = cols or (arr.size // rh)
        mats.append((fn.replace("corr_", "").replace("_ct.bin", ""), arr.reshape(arr.size // c, c)))

    rep = Report(os.path.join(outdir, "report.txt"))
    rep.h(f"Experiment: {name} (input-bit / ciphertext-bit correlation)")
    rep.w(f"NR = {nr}. N(trials) = {N}. Noise-floor sd = 1 / sqrt(N) = {floor:.5f}")

    for lab, M in mats:
        cells = M.size
        chance_max = floor * math.sqrt(2 * math.log(cells))
        beyond = int(np.sum(np.abs(M) > 4 * floor))
        exp_beyond = cells * 2 * stats.norm.sf(4)
        D, pks = stats.kstest((M.ravel() / floor), "norm")

        rep.s(f"{lab} -> ct. Shape = {M.shape[0]}x{M.shape[1]} ({cells} cells)")
        rep.w(f"Mean r = {M.mean():+.6f}. STD r = {M.std():.5f} (floor {floor:.5f})")
        rep.w(f"Max |r| = {np.abs(M).max():.5f}. Chance max ~ {chance_max:.5f}")
        rep.w(f"Cells |r| > 4floor = {beyond} (expected under null ~ {exp_beyond:.1f})")
        rep.w(f"KS versus N(0, 1 / N) = D = {D:.4f}. P = {pks:.4g} -> {"conistent with noise" if pks > 0.05 else "deviates"}")
    rep.close()

    vlim = 4 * floor
    fig, axes = plt.subplots(1, len(mats) + 1, figsize=(4.3 * (len(mats) + 1), 4.3))
    allv = []

    for ax, (lab, M) in zip(axes, mats):
        allv.append(M.ravel())
        im = ax.imshow(block_reduce(M, 768), origin="lower", cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
        ax.set_title(f"{lab} -> ct")
        ax.set_xlabel("Ct bit")
        ax.set_ylabel(f"{lab} bit")
        fig.colorbar(im, ax=ax, fraction=0.046)

    v = np.concatenate(allv)
    sns.histplot(v, bins=200, stat="density", color=BLUE, ax=axes[-1], alpha=0.8)
    xs = np.linspace(-6 * floor, 6 * floor, 300)
    axes[-1].plot(xs, stats.norm.pdf(xs, 0, floor), color=RED, lw=1.3, label=f"N(0, {floor:.3f})")
    axes[-1].set_title(f"All r (max |r| = {np.abs(v).max():.3f})")
    axes[-1].legend(fontsize=8)
    fig.suptitle(f"{name} - correlation heatmaps + distribution (N = {N})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "correlation.png"))
    plt.close(fig)

def hexcol_to_bytes(csv_path, col, maxrows=None):
    s = pd.read_csv(csv_path, usecols=[col], nrows=maxrows)[col].dropna().astype(str)
    n = len(s)
    arr = np.frombuffer(bytes.fromhex("".join(s.tolist())), dtype=np.uint8)
    return arr.reshape(n, -1)

def analyze_differential(exp, name, outdir):
    nrs = sorted(exp["csvs"])
    rows = []
    dist_lownr = {}

    for nr in nrs:
        nr_dir, csvname = exp["csvs"][nr]
        p = os.path.join(nr_dir, csvname)
        wt = pd.read_csv(p, usecols=["out_diff_weight"])["out_diff_weight"].to_numpy()
        od = hexcol_to_bytes(p, "out_diff_hex")
        N = len(od)

        chis = []
        maxp = 0.0

        for b in range(od.shape[1]):
            count = np.bincount(od[:, b], minlength=256).astype(float)
            chis.append(((count - N / 256) ** 2 / (N / 256)).sum() / 255)
            maxp = max(maxp, count.max() / N)

        rows.append((nr, wt.mean(), max(chis), float(np.mean(chis)), maxp, N))
        if nr <= 5:
            dist_lownr[nr] = np.bincount(od[:, 0], minlength=256) / N

        R = pd.DataFrame(rows, columns=["nr", "mean_wt", "chi2_max", "chi2_mean", "max_prob", "N"])

        rep = Report(os.path.join(outdir, "report.txt"))
        rep.h(f"Experiment: {name} (single-block differential, fixed input difference)")
        rep.w(f"NR: {R.nr.min()} to {R.nr.max()}. Pairs / NR ~ {int(R.N.iloc[0])}. Uniform byte probability = 1 / 256 = {1 / 256:.5f}")
        rep.s("Per-NR (output-difference distribution versus uniform)")
        rep.w(f"{"NR":>4} {"mean_wt":>9} {"chi2/255_max":>13} {"chi2/255_mean":>14} {"max_byteprob":>13}")

        for _, r in R.iterrows():
            rep.w(f"{int(r.nr):>4} {r.mean_wt:>9.3f} {r.chi2_max:>13.3f} {r.chi2_mean:>14.3f} {r.max_prob:>13.5f}")

        coll = next((int(r.nr) for _, r in R.iterrows() if r.chi2_max < 2.0), None)
        rep.s("Interpretation")
        rep.w(f"Differential structure (chi2 / 255) collapses to ~1 (uniform) by NR = {coll}")
        rep.close()

        fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
        ax[0].plot(R.nr, R.mean_wt, marker="o", ms=4, color=BLUE)
        ax[0].axhline(64, ls="--", color="gray", lw=1, label="64 = 50%")
        ax[0].set_title("Mean output-difference weight")
        ax[0].set_xlabel("NR")
        ax[0].set_ylabel("Bits")
        ax[0].legend(fontsize=8)

        ax[1].semilogy(R.nr, R.chi2_max, marker="o", ms=4, color=BLUE, label="Max over bytes")
        ax[1].semilogy(R.nr, R.chi2_mean, marker="s", ms=3, color=ORANGE, label="Mean over bytes")
        ax[1].axhline(1.0, ls="--", color="gray", lw=1, label="uniform")
        ax[1].axvline(4, ls=":", color="RED", lw=1, label="Wide-trail NR = 4")
        ax[1].set_title("Output-byte-difference chi2 / 255 (log)")
        ax[1].set_xlabel("NR")
        ax[1].legend(fontsize=8)

        if dist_lownr:
            lows = sorted(dist_lownr)
            M = np.vstack([dist_lownr[n] for n in lows])

            im = ax[2].imshow(M, aspect="auto", origin="lower", cmap="magma", extent=[0, 256, lows[0] - 0.5, lows[-1] + 0.5])
            ax[2].set_yticks(lows)
            ax[2].set_title("Byte-0 output-diff distribution")
            ax[2].set_xlabel("Byte-diff value")
            ax[2].set_ylabel("NR")
            fig.colorbar(im, ax=ax[2], label="Probability")
        fig.suptitle(f"{name} - differential distribution collapse", fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "differential.png"))
        plt.close(fig)

def analyze_integral(exp, name, outdir):
    nrs = sorted(exp["csvs"])
    rows = []
    for nr in nrs:
        nr_dir, csvname = exp["csvs"][nr]
        df = pd.read_csv(os.path.join(nr_dir, csvname), usecols=["balanced_bytes", "fully_balanced"])
        rows.append((nr, df["balanced_bytes"].mean(), df["balanced_bytes"].std(), df["fully_balanced"].mean(), len(df)))
    R = pd.DataFrame(rows, columns=["nr", "mean_bal", "std_bal", "frac_full", "N"])
    chance = 16 / 256.0

    rep = Report(os.path.join(outdir, "report.txt"))
    rep.h(f"Experiment: {name} (integral / square property)")
    rep.w(f"NR {R.nr.min()} to {R.nr.max()}. Lambda-sets / NR ~ {int(R.N.iloc[0])}. Chance balanced bytes = 16 / 256 = {chance:.4f}")
    rep.s("Per-NR (balance of output-byte XOR over Lambda-set)")
    rep.w(f"{"NR":>4} {"mean_balanced":>14} {"std":>7} {"frac_fully_balanced":>20}")

    for _, r in R.iterrows():
        rep.w(f"{int(r.nr):>4} {r.mean_bal:>14.4f} {r.std_bal:>7.3f} {r.frac_full:>20.5f}")        

    brk = next((int(r.nr) for _, r in R.iterrows() if r.frac_full < 0.5), None)
    rep.s("Interpretation")
    rep.w(f"Property holds (16 / 16 balanced, frac = 1) through NR <= {brk - 1}. Breaks at NR = {brk}")
    rep.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].plot(R.nr, R.mean_bal, marker="o", ms=5, color=BLUE)
    ax[0].axhline(16, ls="--", color=GREEN, lw=1, label="16 = balanced")
    ax[0].axhline(chance, ls="--", color="gray", lw=1, label=f"Chance {chance:.3f}")
    ax[0].axvline(4, ls=":", color=RED, lw=1, label="Break NR = 4")
    ax[0].set_title("Mean balanced output bytes")
    ax[0].set_xlabel("NR")
    ax[0].set_ylabel("Baanced bytes (of 16)")
    ax[0].legend(fontsize=8)

    ax[1].plot(R.nr, R.frac_full, marker="o", ms=5, color=BLUE)
    ax[1].axvline(4, ls=":", color=RED, lw=1)
    ax[1].set_title("Fraction of fully-balanced Lambda-sets")
    ax[1].set_xlabel("NR")
    ax[1].set_ylabel("Fraction")
    ax[1].set_ylim(-0.05, 1.05)

    fig.suptitle(f"{name} - integral (square) property versus rounds", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "integral_square.png"))
    plt.close(fig)

def main():
    exps = experiments("./")
    if not exps:
        return 

    for exp_dir, exp in sorted(exps.items()):
        name = os.path.basename(exp_dir.rstrip("/\\")) or "root"
        kind = classify(exp)

        outdir = os.path.join("./reports", name)
        os.makedirs(outdir, exist_ok=True)

        anrs = sorted(exp["csvs"])
        try:
            if kind in ("scalar", "nrnext", "perbit"):
                analyze_scalar(exp, name, outdir, 4096, 4)
            elif kind == "matrix":
                analyze_matrix(exp, name, outdir, None)
            elif kind == "correlation":
                analyze_correlation(exp, name, outdir, None)
            elif kind == "differential":
                analyze_differential(exp, name, outdir)
            elif kind == "integral":
                analyze_integral(exp, name, outdir)
            else:
                continue
            analyze_raw(exp, outdir, 256)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(e)

main()
