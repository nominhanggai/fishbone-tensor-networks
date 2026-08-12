"""User-friendly interface for the multi-bath *fishbone* geometry.

A fishbone is a 1D chain of electronic sites, each carrying its own bath (a
"comb") or two baths -- one on each side of the site -- giving the fishbone its
name::

        bath  bath  bath                 (right / "vb" baths, optional)
          |     |     |
    ...--E0----E1----E2--...             (electronic backbone)
          |     |     |
        bath  bath  bath                 (left / "eb" baths)

This wraps the low-level :class:`~fishbonett.model.FishBoneH` Hamiltonian
builder and the :class:`~fishbonett.fishbone.FishBoneNet` tree-TEBD engine
behind a single declarative class, so a fishbone simulation is specified and
run without hand-writing the comb sweep::

    import numpy as np
    from fishbonett.fishbone_sim import Fishbone
    from fishbonett.simulate import Bath
    from fishbonett.stuff import sigma_x, sigma_z

    J = lambda w: 0.2 * w * np.exp(-w / 5)
    bath = lambda: Bath(J=J, domain=(0, 40), n_modes=20, phys_dim=8,
                        coupling=sigma_z)
    fb = Fishbone(sites=[0.5 * sigma_z + sigma_x] * 3,     # 3 electronic sites
                  baths=[bath(), bath(), bath()],          # one bath each
                  backbone=[0.4 * np.kron(sigma_z, sigma_z)] * 2)  # e-e coupling
    res = fb.run(dt=0.02, t_max=2.0, bond_dim=60,
                 observables={'sz': sigma_z})
    res.expect['sz']        # shape (n_steps, n_sites)
    res.rdm                 # shape (n_steps, n_sites, d, d)
"""
import numpy as np

from fishbonett.model import FishBoneH
from fishbonett.fishbone import FishBoneNet
from fishbonett.stuff import sigma_x, sigma_z
from fishbonett.simulate import Result

__all__ = ["Fishbone"]


def _as_pair(entry):
    """Normalise a per-site bath spec to (left_bath, right_bath)."""
    if entry is None:
        return (None, None)
    if isinstance(entry, (tuple, list)):
        if len(entry) == 1:
            return (entry[0], None)
        if len(entry) == 2:
            return (entry[0], entry[1])
        raise ValueError("a site takes at most two baths (left, right)")
    return (entry, None)


class Fishbone:
    """A 1D chain of electronic sites, each coupled to one or two baths.

    Parameters
    ----------
    sites : list of (d, d) array
        Electronic site Hamiltonians, one per site.
    baths : list
        One entry per site.  Each entry is a single :class:`~fishbonett.simulate.Bath`
        (a left/``eb`` bath), a ``(left, right)`` pair of baths (fishbone with two
        baths per site), or ``None``.  Each :class:`Bath` may carry its own
        ``coupling`` operator; it defaults to ``sigma_z`` for a left bath and
        ``sigma_x`` for a right bath.
    backbone : list of (d*d, d*d) array, optional
        Nearest-neighbour couplings between electronic sites ``i`` and ``i+1``
        (length ``n_sites - 1``).  Default: uncoupled sites.

    Notes
    -----
    All baths must share the same frequency ``domain`` (a limitation of the
    underlying comb Hamiltonian).  The site Hilbert-space dimension ``d`` may
    differ between sites; ``backbone[i]`` must then act on ``d_i * d_{i+1}``.
    """

    def __init__(self, sites, baths, backbone=None):
        self.sites = [np.asarray(h, complex) for h in sites]
        self.nc = len(self.sites)
        if len(baths) != self.nc:
            raise ValueError("baths must have one entry per site")
        self.baths = [_as_pair(b) for b in baths]
        self.de = [h.shape[0] for h in self.sites]
        if backbone is None:
            backbone = [np.zeros((self.de[i] * self.de[i + 1],) * 2, complex)
                        for i in range(self.nc - 1)]
        if len(backbone) != max(self.nc - 1, 0):
            raise ValueError("backbone must have n_sites - 1 entries")
        self.backbone = [np.asarray(b, complex) for b in backbone]
        self._check_domain()

    def _check_domain(self):
        doms = []
        for lb, rb in self.baths:
            for b in (lb, rb):
                if b is not None:
                    doms.append(tuple(b.domain))
        if doms and len(set(doms)) != 1:
            raise ValueError("all baths must share the same frequency domain")
        self.domain = list(doms[0]) if doms else [0.0, 1.0]

    # -- Hamiltonian assembly ------------------------------------------------
    def _build_pd(self):
        pd = np.empty([self.nc, 4], dtype=object)
        for n, ((lb, rb), de) in enumerate(zip(self.baths, self.de)):
            pd[n, 0] = [lb.phys_dim] * lb.n_modes if lb is not None else []
            pd[n, 1] = [de]
            pd[n, 2] = []
            pd[n, 3] = [rb.phys_dim] * rb.n_modes if rb is not None else []
        return pd

    def _discretizer(self):
        discs = set()
        for lb, rb in self.baths:
            for b in (lb, rb):
                if b is not None:
                    discs.add(b.discretization)
        if len(discs) > 1:
            raise ValueError("all baths must use the same discretization")
        # any bath carries the (shared) discretizer factory
        for lb, rb in self.baths:
            for b in (lb, rb):
                if b is not None:
                    return b.discretizer()
        return None

    def _build_h(self, g, ncap):
        pd = self._build_pd()
        eth = FishBoneH(pd)
        eth.domain = self.domain
        he_dy, hv_dy, h1e, h1v = [], [], [], []
        for n, ((lb, rb), de, hsite) in enumerate(
                zip(self.baths, self.de, self.sites)):
            two_bath = rb is not None
            if lb is not None:
                eth.sd[n, 0] = lb.spectral_density()
                lc = lb.coupling if lb.coupling is not None else sigma_z
                he_dy.append(np.asarray(lc, complex))
            else:
                he_dy.append(np.eye(de))
            if rb is not None:
                eth.sd[n, 1] = rb.spectral_density()
                rc = rb.coupling if rb.coupling is not None else sigma_x
                hv_dy.append(np.asarray(rc, complex))
            else:
                hv_dy.append(np.eye(de))
            # Route the site energy: two-bath sites carry it on h1v (the vb bond;
            # the comb builder's two-bath branch skips h1e), one-bath sites on h1e
            # (the eb bond for a single site, or the backbone bond for a chain).
            if two_bath:
                h1e.append(np.zeros((de, de), complex))
                h1v.append(hsite)
            else:
                h1e.append(hsite)
                h1v.append(np.zeros((de, de), complex))
        eth.he_dy = he_dy
        eth.hv_dy = hv_dy
        eth.h1e = h1e
        eth.h1v = h1v
        if self.nc > 1:
            eth.h2ee = list(self.backbone)
        eth.build(g=g, ncap=ncap, discretizer=self._discretizer())
        return eth

    # -- initial comb state --------------------------------------------------
    def _init_net(self, pd, initial):
        def g_state(dim):
            t = np.zeros(dim, complex)
            t[(0,) * len(dim)] = 1.0
            return t

        nc = self.nc
        eb_t = [[g_state([1, d, 1]) for d in pd[i, 0]] for i in range(nc)]
        e_t = [[g_state([1, d, 1, 1, 1]) for d in pd[i, 1]] for i in range(nc)]
        v_t = [[g_state([1, d, 1]) for d in pd[i, 2]] for i in range(nc)]
        vb_t = [[g_state([1, d, 1]) for d in pd[i, 3]] for i in range(nc)]
        eb_s = [[np.ones(1) for _ in ch] for ch in eb_t]
        e_s = [[np.ones(1) for _ in ch] for ch in e_t]
        v_s = [[np.ones(1) for _ in ch] for ch in v_t]
        vb_s = [[np.ones(1) for _ in ch] for ch in vb_t]
        main_s = [[np.ones(1)] for _ in range(nc)]
        vb_and_main = [vb_s[i] + main_s[i] for i in range(nc)]
        for i in range(nc):
            vec = self._initial_vec(initial, i)
            e_t[i][0][:] = 0.0
            for a in range(len(vec)):
                e_t[i][0][0, a, 0, 0, 0] = vec[a]
        return FishBoneNet((eb_t, e_t, v_t, vb_t), (eb_s, e_s, v_s, vb_and_main))

    def _initial_vec(self, initial, i):
        de = self.de[i]
        if initial is None or (isinstance(initial, str) and initial == "up"):
            v = np.zeros(de, complex); v[0] = 1.0
            return v
        if isinstance(initial, str) and initial == "down":
            v = np.zeros(de, complex); v[min(1, de - 1)] = 1.0
            return v
        if isinstance(initial, str) and initial == "ground":
            w, U = np.linalg.eigh(self.sites[i])
            return U[:, int(np.argmin(w))].astype(complex)
        item = initial[i] if isinstance(initial, (list, tuple)) else initial
        v = np.asarray(item, complex)
        return v / np.linalg.norm(v)

    # -- driver --------------------------------------------------------------
    def run(self, *, dt, t_max=None, n_steps=None, bond_dim=100, trunc_eps=1e-10,
            observables=None, initial="up", g=1.0, ncap=20000):
        """Propagate the fishbone and return a :class:`~fishbonett.simulate.Result`.

        The returned ``Result`` carries per-site data: ``expect[name]`` has shape
        ``(n_steps, n_sites)`` and ``rdm`` has shape ``(n_steps, n_sites, d, d)``.
        ``observables`` maps names to single-site electronic operators (default
        ``sigma_z``/``sigma_x`` when the sites are two-level).
        """
        if n_steps is None:
            if t_max is None:
                raise ValueError("provide either t_max or n_steps")
            n_steps = int(round(t_max / dt))
        if observables is None:
            observables = {"sz": sigma_z, "sx": sigma_x} if all(
                d == 2 for d in self.de) else {}
        eth = self._build_h(g, ncap)
        pd = self._build_pd()
        etn = self._init_net(pd, initial)
        etn.U = eth.get_u(dt=dt)

        n_bond = [eth._L[cn] - 1 for cn in range(self.nc)]
        e_index = [eth._ebL[cn] for cn in range(self.nc)]
        rdms = np.empty((n_steps, self.nc), dtype=object)
        for tn in range(n_steps):
            if self.nc > 1:
                for i in range(self.nc - 1):
                    etn.update_bond(-1, i, bond_dim, trunc_eps)
            for cn in range(self.nc):
                for j in range(n_bond[cn]):
                    etn.update_bond(cn, j, bond_dim, trunc_eps)
            for cn in range(self.nc):
                th = etn.get_theta1(cn, e_index[cn])
                rho = np.einsum("LiUDR,LjUDR->ij", th, th.conj())
                rdms[tn, cn] = rho / np.trace(rho).real

        expect = {}
        for name, O in observables.items():
            O = np.asarray(O)
            expect[name] = np.array(
                [[np.trace(rdms[tn, cn] @ O).real for cn in range(self.nc)]
                 for tn in range(n_steps)])
        rdm = np.array([[rdms[tn, cn] for cn in range(self.nc)]
                        for tn in range(n_steps)])
        t = np.arange(1, n_steps + 1) * dt
        return Result(t=t, expect=expect, rdm=rdm, method="fishbone",
                      meta={"n_sites": self.nc})
