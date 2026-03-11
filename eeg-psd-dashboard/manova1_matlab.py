import numpy as np
from scipy.linalg import cholesky, solve_triangular, eigh
from scipy.stats import chi2

def manova1_like_matlab(X, group, alpha=0.05):
    X = np.asarray(X, dtype=float)
    g = np.asarray(list(group), dtype=object).reshape(-1)

    # remove NaNs em X
    no_nan_X = ~np.isnan(X).any(axis=1)
    is_nan_mask_original = ~no_nan_X
    X2 = X[no_nan_X, :]
    g2 = g[no_nan_X]

    # grp2idx estável (ordem de aparição)
    label_to_idx = {}
    group_names = []
    group_idx = np.empty(X2.shape[0], dtype=int)
    for i, lab in enumerate(g2):
        if lab not in label_to_idx:
            label_to_idx[lab] = len(group_names)
            group_names.append(lab)
        group_idx[i] = label_to_idx[lab]

    nsample, nvar = X2.shape
    ngroups = len(group_names)

    xm = X2.mean(axis=0)
    x_centered = X2 - xm
    TSSP = x_centered.T @ x_centered

    WSSP = np.zeros((nvar, nvar), dtype=float)
    for j in range(ngroups):
        rows = np.where(group_idx == j)[0]
        if rows.size > 1:
            gx = x_centered[rows, :]
            gx = gx - gx.mean(axis=0)
            WSSP += gx.T @ gx

    BSSP = TSSP - WSSP

    # Regularize WSSP if it's not positive definite (singular or ill-conditioned)
    # This prevents 'cholesky' and min_eigen <= -1 errors for high-dimensional or collinear data (like PSD)
    try:
        R = cholesky(WSSP, lower=False, check_finite=False)
    except Exception:
        # Apply Ridge penalty relative to the matrix trace to safely guarantee positive definiteness
        reg = max(1e-6 * (np.trace(WSSP) / nvar), 1e-8)
        WSSP_reg = WSSP + np.eye(nvar) * reg
        R = cholesky(WSSP_reg, lower=False, check_finite=False)

    S = solve_triangular(R.T, BSSP, lower=True, check_finite=False)
    S = solve_triangular(R, S.T, lower=False, check_finite=False).T
    S = 0.5 * (S + S.T)

    evals, evecs = eigh(S, check_finite=False)  # asc
    e = evals
    vv = evecs

    v = solve_triangular(R, vv, lower=False, check_finite=False)

    ei = np.argsort(e)
    e = e[ei]
    v = v[:, ei]
    
    # Clip small negative numerical artifacts to slightly above 0 to prevent log(lambda) errors down the line
    e = np.where(e < -0.999, -0.999, e)

    maxdim = min(ngroups - 1, nvar)
    dims = np.arange(0, maxdim, dtype=int)

    lam_all = np.flip(1.0 / np.cumprod(e + 1.0))
    lam = lam_all[dims]

    chistat = -(nsample - 1.0 - (ngroups + nvar)/2.0) * np.log(lam)
    chisqdf = (nvar - dims) * (ngroups - 1 - dims)
    pp = 1.0 - chi2.cdf(chistat, chisqdf)

    idx_ok = np.where(pp > alpha)[0]
    d = int(dims[idx_ok[0]]) if idx_ok.size > 0 else int(dims.max() + 1)

    # reorder DESC
    e_desc = np.flip(e)
    v_desc = v[:, np.flip(np.arange(v.shape[1]))]

    # rescale so within-group var = 1
    vs = np.diag(v_desc.T @ WSSP @ v_desc) / (nsample - ngroups)
    vs = np.where(vs <= 0, 1.0, vs)
    v_desc = v_desc / np.sqrt(vs)[None, :]

    # flip sign for consistency
    neg = (v_desc.sum(axis=0) < 0)
    v_desc[:, neg] *= -1

    canon = x_centered @ v_desc

    gmean = np.full((ngroups, canon.shape[1]), np.nan, dtype=float)
    for j in range(ngroups):
        rows = np.where(group_idx == j)[0]
        gmean[j, :] = canon[rows, :].mean(axis=0)

    mdist = np.sum((canon - gmean[group_idx, :])**2, axis=1)
    diff = gmean[:, None, :] - gmean[None, :, :]
    gmdist = np.sum(diff**2, axis=2)

    # reinsert NaNs
    if np.any(is_nan_mask_original):
        canon_full = np.full((X.shape[0], canon.shape[1]), np.nan, dtype=float)
        mdist_full = np.full((X.shape[0],), np.nan, dtype=float)
        kept_rows = np.where(~is_nan_mask_original)[0]
        canon_full[kept_rows, :] = canon
        mdist_full[kept_rows] = mdist
        canon = canon_full
        mdist = mdist_full

    stats = {
        "W": WSSP, "B": BSSP, "T": TSSP,
        "dfW": int(nsample - ngroups),
        "dfB": int(ngroups - 1),
        "dfT": int(nsample - 1),
        "lambda": lam,
        "chisq": chistat,
        "chisqdf": chisqdf,
        "eigenval": e_desc,
        "eigenvec": v_desc,
        "canon": canon,
        "mdist": mdist,
        "gmdist": gmdist,
        "gnames": group_names
    }

    return d, pp, stats
