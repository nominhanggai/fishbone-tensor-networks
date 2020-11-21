import numpy as np
import sys


class SimpleTTPS:
    """ Simple Tree Like Tensor-Product States
        
                         ⋮
        --d33--d32--d31--E3--V3--b31--b32--b33--
                         |
        --d23--d22--d21--E2--V2--b21--b22--b23--
                         |
        --d13--d12--d11--E1--V1--b11--b12--b13--
                         |
        --d03--d02--d01--E0--V0--b01--b02--b03--
                         ⋮ 
    """

    def __init__(self, B, S,  # ele_list, vib_list, e_bath_list, v_bath_list
                 ):
        # self.e = ele_list
        # self.v = vib_list
        # self.eb = e_bath_list
        # self.vb = v_bath_list
        (eb, ev, vb) = B
        (eb_s, ev_s, vb_s) = S

        try:
            assert len(eb) == len(ev) == len(vb)
        except AssertionError:
            print("Lengths of the tensor lists don't match. "
                  "Look back the diagram in the comment of the class SimpleTTPS. "
                  "The tensors in the lists should compose a comb-like network.",
                  file=sys.stderr
                  )
            raise
            sys.exit(1)

        self._nc = len(ev)
        self._evL = [len(ev[n]) for n in range(self._nc)]
        self._ebL = [len(eb[n]) for n in range(self._nc)]
        self._vbL = [len(vb[n]) for n in range(self._nc)]
        self._L = [sum(x) for x in zip(self._ebL, self._evL, self._vbL)]

        self.ttnB = []
        # ttn means tree tensor network. The self.ttn array stores the tensors in the whole network.
        for i in range(self._nc):
            self.ttnB.append(
                eb[i] + ev[i] + vb[i]
            )
        self._pD = []  # dimensions of physical legs
        for n in range(self._nc):
            self._pD.append(
                [self.ttnB[n][i].shape[1] for i in range(self._L[i])]
            )
        self.ttnS = []

        """ ttnS = [ [ S_11, S_12, ..., S_1M, S_backbone_1 ],
                     [ S_21, S_22, ..., S_2N, S_backbone_2 ],
                     ...
                   ]
        """
        for n in range(self._nc):
            self.ttnS.append(eb_s[n] + ev_s[n] + vb_s[n])

        self.ttnH = []
        """ ttnH = [ [ H_11, H_12, ..., H_1M, H_backbone_1 ],
                     [ H_21, H_22, ..., H_2N, H_backbone_2 ],
                     ...
                   ]
        """
        for n in range(self._nc):
            self.ttnH.append(
                [np.empty(0, dtype=np.float) for i in range(self._L[n])]
            )

    def get_theta1(self, n, i):
        """
        Calculate effective single-site wave function on sites i in B canonical form.
        :param n: Which chain
        :param i: Which site
        :return: S*B
        """
        assert n <= len(self._ebL) - 1
        for j in range(self._nc):
            assert 0 <= i < self._ebL[n] + self._evL[n] + self._vbL[n]
        return np.tensordot(
            np.diag(self.ttnS[n][i]),
            self.ttnB[n][i],
            [1, 0]
        )  # vL [vL'], [vL] i vU vD  vR -> vL i vU vD vR

    def get_theta2(self, n, i):
        """Calculate effective two-site wave function on sites i,j=(i+1) in mixed canonical form.
        """
        # n=-1 means the backbone chain. When n=-1, i is the bond number bottom up
        if n == -1:
            try:
                assert 0 <= i < self._nc - 1
            except AssertionError:
                print("The bond is out of the range.",
                      file=sys.stderr
                      )
                raise
                sys.exit(1)
            upward_b = np.tensordot(
                self.get_theta1(i, self._ebL[i]),
                np.diag(self.ttnS[i][self._ebL[i]] ** (-1))
            )  # vL i [vU] vD vR, vU [vD] -> vL i vD vR vU
            return np.tensordot(
                upward_b,
                self.get_theta1(i + 1, self._ebL[i + 1]),
                [2, 4]
            )  # vL i [vU] vD vR , vL' j vU' [vD'] vR' -> {vL i VD vR; VL' j vU' vR'}

        if n != -1:
            assert n <= self._nc - 1
            assert 0 <= i < self._ebL[i] + self._vbL[i] + 1
            return np.tensordot(
                self.get_theta1(n, i), self.ttnB[n][i + 1], axes=1
            )  # vL i _vU_ _vD_ [vR],  [vL] j _vU_ _vD_ vR -> {vL i _vU_ _vD_; j _vU_ _vD_ vR}

    def split_truncate_theta(self, theta, n, i):
        """
        TODO
        Split the contracted two-site wave function and truncate the number of singular values.
        :param theta: the contracted two-site wave function
        :param n: which chain
        :param i: which bond on the chain
        :return: B_1, S, B_2
        """
        pass

    def update_bond(self):
        """TODO
        """
        pass


def init_ttn(nc, L, d1, d2):
    eb = np.zeros([1, d1, 1], np.float)  # vL i vR
    eb[0, 0, 0] = 1.
    ebs = [eb.copy() for i in range(L)]
    ebss = [ebs.copy() for i in range(nc)]
    #
    ev = np.zeros([1, d1, 1, 1, 1], np.float)  # vL i vU vD vR
    ev[0, 0, 0, 0, 0] = 1.
    evs = [ev.copy() for i in range(2)]
    evss = [evs.copy() for i in range(nc)]
    #
    vb = np.zeros([1, d1, 1], np.float)  # vL i vR
    vb[0, 0, 0] = 1.
    vbs = [vb.copy() for i in range(L)]
    vbss = [vbs.copy() for i in range(nc)]

    eb_s = np.ones([1], np.float)
    eb_ss = [eb_s.copy() for i in range(L)]
    eb_sss = [eb_ss.copy() for i in range(nc)]

    ev_s = np.ones([1], np.float)
    ev_ss = [ev_s.copy() for i in range(2)]
    ev_sss = [ev_ss.copy() for i in range(nc)]

    vb_s = np.ones([1], np.float)
    vb_ss = [vb_s.copy() for i in range(L)]
    vb_sss = [vb_ss.copy() for i in range(nc)]
    # print(ebss[1])
    # print(evss[1])
    # print(vbss[1])
    return SimpleTTPS(
        (ebss, evss, vbss),
        (eb_sss, ev_sss, vb_sss)
    )


if __name__ == "__main__":
    ttn = init_ttn(nc=2, L=3, d1=5, d2=6)
    ttn.get_theta2(-1, 1)
