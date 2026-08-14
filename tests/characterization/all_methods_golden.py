"""Golden capture/check across **every** method in the registry.

The gate for structural refactors of the model/representation/propagator layers: a
restructure must not move a number.  Runs each ``(model, method)`` pair the registry
declares on a small fixed problem and records ``(t, rdm, expect, max_bond)``.

Not collected by pytest (no ``test_`` prefix) -- it is a two-step tool, and the
baseline is environment-specific rather than something to commit::

    python tests/characterization/all_methods_golden.py capture ref.pkl
    ...make the change...
    python tests/characterization/all_methods_golden.py check ref.pkl

``check`` exits non-zero if anything moved and prints which field differs, which is
what tells an intended change apart from an accident.  It is strictly stronger than
the unit suite for this purpose: it compares exactly (``==``, not ``allclose``), so
it catches a reordered Trotter term that stays within tolerance.

Deliberate changes are expected sometimes -- re-capture and say so in the commit.

.. note::
   A baseline is specific to the contraction backend it was captured on.  Comparing
   an ``opt_einsum`` capture against a ``FISHBONETT_EINSUM=numpy`` run shows one
   method differing: ``chain/trotter-mpo``, by 2.2e-16 in the RDM (under one machine
   epsilon, and 0 in the observable).  That is a single rounding step from a
   different contraction order, not a disagreement -- but since this file compares
   exactly, capture and check on the same backend.
"""
import pickle
import sys
import traceback

import numpy as np

from fishbonett import Bath, Fishbone, SystemBath
from fishbonett.models import TreeFishbone
from fishbonett.models import registry as R
from fishbonett.operators import sigma_x, sigma_z

_J = lambda w: 0.2 * w * np.exp(-w / 5.0)


def _bath():
    return Bath(J=_J, domain=(0.0, 40.0), n_modes=3, phys_dim=4)


#: Old key -> new, for comparing a baseline captured before the taxonomy was
#: re-axed.  ``chain``/``star`` were half of a *representation* and ``mode-tree`` a state
#: *geometry*; all three are the one ``system-bath`` model.  The old
#: ``tree-tebd-static`` label on the multichannel model became
#: ``schrodinger-star-tree-tebd``: same engine, but a
#: different **representation** (schrodinger-star, where the multi-site models are
#: schrodinger-chain), which one row could not carry.
#:
#: The runs are the same runs either way, so every number must still match
#: exactly; only the labels moved.
_RENAMED = {"chain": "system-bath", "star": "system-bath",
            "mode-tree": "system-bath"}
_RENAMED_METHOD = {
    ("multichannel", "tree-tebd-static"): "schrodinger-star-tree-tebd",
}


def _model_for(key):
    h = 0.5 * sigma_x
    if key == "system-bath":
        return SystemBath(h=h, coupling=sigma_z, bath=_bath())
    if key == "multichannel":
        mc = Bath(J=[_J, _J], coupling=[sigma_z, sigma_x], domain=(0.0, 40.0),
                  n_modes=3, phys_dim=4)
        return SystemBath(h=h, coupling=[sigma_z, sigma_x], bath=mc)
    if key == "comb":
        return Fishbone(sites=[h, h], baths=[_bath(), None])
    if key == "site-tree":
        return TreeFishbone(sites=[h], edges=[], baths=[_bath()])
    raise AssertionError(key)


def _fixed_bond():
    """Methods needing an explicit cap.  Read from wherever it currently lives so
    the harness survives the move of this table into the registry."""
    for mod, name in (("fishbonett.models.registry", "FIXED_BOND_METHODS"),
                      ("fishbonett.models.system_bath", "_FIXED_BOND_METHODS")):
        try:
            return getattr(__import__(mod, fromlist=[name]), name)
        except AttributeError:
            continue
    raise RuntimeError("cannot find the fixed-bond method table")


def capture():
    fixed = _fixed_bond()
    out = {}
    for model_key in sorted(R.MODELS):
        for method in R.methods_of(model_key):
            obj = _model_for(model_key)          # fresh instance per run
            kw = dict(dt=0.02, n_steps=3, observables={"sz": sigma_z},
                      trunc_eps=1e-7)
            if method in fixed:
                kw["bond_dim"] = 12
            key = (model_key, method)
            try:
                if (model_key == "multichannel"
                        and method == R.SCHRODINGER_STAR_TREE_TEBD):
                    r = obj.run(**kw)            # selected by the bath, not by name
                else:
                    r = obj.run(method=method, **kw)
                out[key] = dict(t=r.t, rdm=r.rdm, expect=dict(r.expect),
                                max_bond=r.max_bond, method=r.method)
            except Exception as e:               # record the failure verbatim too
                out[key] = dict(error=f"{type(e).__name__}: {e}")
                traceback.print_exc()
    return out


def same(a, b, path=""):
    """Exact equality, tolerant of None / NaN / object arrays / dicts."""
    if a is None or b is None:
        return (a is None) == (b is None)
    if isinstance(a, dict) or isinstance(b, dict):
        if set(a) != set(b):
            return False
        return all(same(a[k], b[k], f"{path}.{k}") for k in a)
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    aa, bb = np.asarray(a), np.asarray(b)
    if aa.shape != bb.shape:
        return False
    if aa.dtype == object or bb.dtype == object:
        return all(same(x, y, path) for x, y in zip(aa.ravel(), bb.ravel()))
    if np.issubdtype(aa.dtype, np.floating) or np.issubdtype(aa.dtype, np.complexfloating):
        both_nan = np.isnan(aa) & np.isnan(bb)
        return bool(np.all(both_nan | (aa == bb)))
    return bool(np.array_equal(aa, bb))


def _relabel(ref):
    """Apply :data:`_RENAMED` / :data:`_RENAMED_METHOD` to an old baseline.

    Only labels are rewritten -- the recorded arrays are untouched -- so a pass
    still means every number is bit-for-bit what it was, which is the whole point
    of comparing against a pre-rename capture.
    """
    out = {}
    for (mk, meth), v in ref.items():
        new_meth = _RENAMED_METHOD.get(
            (mk, meth), R._RENAMED_METHODS.get(meth, meth))
        v = dict(v)
        if v.get("method") == meth:
            v["method"] = new_meth
        out[(_RENAMED.get(mk, mk), new_meth)] = v
    return out


def check(ref):
    cur = capture()
    ref = _relabel(ref)
    if set(ref) != set(cur):
        print("METHOD SET CHANGED")
        for k in sorted(set(ref) - set(cur)):
            print("  gone:", k)
        for k in sorted(set(cur) - set(ref)):
            print("  new :", k)
    ok = True
    for key in sorted(set(ref) & set(cur)):
        r, c = ref[key], cur[key]
        if "error" in r or "error" in c:
            match = r.get("error") == c.get("error")
        else:
            match = all(same(r[f], c[f], f"{key}.{f}")
                        for f in ("t", "rdm", "expect", "max_bond", "method"))
        ok &= match
        print(f"{'OK  ' if match else 'MOVED'}  {key[0]:<13} {key[1]}"
              + ("" if match else "   <-- NUMBERS CHANGED"))
        if not match and "error" not in r and "error" not in c:
            for f in ("t", "rdm", "expect", "max_bond", "method"):
                if not same(r[f], c[f]):
                    print(f"         field {f!r} differs: {r[f]!r} vs {c[f]!r}"[:300])
    print()
    print("ALL METHODS BIT-FOR-BIT UNCHANGED:", ok)
    return ok


if __name__ == "__main__":
    mode, path = sys.argv[1], sys.argv[2]
    if mode == "capture":
        data = capture()
        with open(path, "wb") as f:
            pickle.dump(data, f)
        nerr = sum(1 for v in data.values() if "error" in v)
        print(f"captured {len(data)} (model, method) runs; {nerr} raised")
        for k, v in sorted(data.items()):
            print(f"  {k[0]:<13} {k[1]:<18} "
                  + (v["error"][:60] if "error" in v else
                     f"rdm{np.shape(v['rdm'])} max_bond={v['max_bond']}"))
    else:
        with open(path, "rb") as f:
            ref = pickle.load(f)
        sys.exit(0 if check(ref) else 1)
