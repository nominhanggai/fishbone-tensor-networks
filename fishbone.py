import numpy as np
import sys
from scipy.linalg import svd


class SimpleTTPS:
    """ Simple Tree Like Tensor-Product States
                         ⋮
        --d33--d32--d31--E3--V3--b31--b32--b33--  3
                         |
        --d23--d22--d21--E2--V2--b21--b22--b23--  2
                         |
        --d13--d12--d11--E1--V1--b11--b12--b13--  1
                         |
        --d03--d02--d01--E0--V0--b01--b02--b03--  0
                         ⋮
                        -1
    Let's look at what's already in the diagram and ignore the ellipsis.
    Bond d03--d02 will be the bond 0 on the chain 0.
    Bond d12--d11 will be the bond 1 on the china 1.
    Bond E0-E1 will be the bond 0 on the chain -1.

    """

    def __init__(self, B, S, H # ele_list, vib_list, e_bath_list, v_bath_list
                 ):
        """

        :param B:
        :type B:
        :param S:
        :type S:
        """
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
        for n in range(self._nc):
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
            if self._evL[n] == 2:
                H_eb, H_ev, H_vb= H[n]
            self.ttnH.append(h_ev + H_ev + H_vb)
            elif self._evL[n] == 1:
                H_eb, H_ev = H[n]
                self.ttnH.append(H_ev + H_ev)

    def get_theta1(self, n: int, i: int):
        """
        Calculate effective single-site wave function on sites i in B canonical form.
        :return:
        :rtype:
        :param n: Which chain
        :param i: Which site
        :return: S*B
        """
        assert -1 <= n <= len(self._ebL) - 1
        assert 0 <= i < self._L[n] - 1
        return np.tensordot(
            np.diag(self.ttnS[n][i]),
            self.ttnB[n][i],
            [1, 0]
        )  # vL [vL'], [vL] i vU vD  vR -> vL i vU vD vR

    def get_theta2(self, n: int, i: int):
        """Calculate effective two-site wave function on sites i,j=(i+1) in mixed canonical form.
        :param n:
        :type n: int
        :param i:
        :type i:
        :return:
        :rtype:
        """
        # n=0 means the backbone chain. When n=0, i is the bond number bottom up
        if n == -1:
            try:
                assert 0 <= i <= self._nc - 2
            except AssertionError:
                print("The bond is out of the range.",
                      file=sys.stderr
                      )
                raise
                sys.exit(1)
            upward_b = np.tensordot(
                self.get_theta1(i, self._ebL[i]),
                np.diag(self.ttnS[i][self._ebL[i]] ** (-1)),
                [2, 0]
            )  # vL i [vU] vD vR, [vU] vU -> vL i vD vR vU
            print(upward_b.shape)
            upward_b = np.transpose(upward_b, [0, 1, 4, 2, 3])  # vL i vD vR vU -> vL i vU vD vR
            return np.tensordot(
                upward_b,
                self.get_theta1(i + 1, self._ebL[i + 1]),
                [2, 4]
            )  # vL i [vU] vD vR , vL' j vU' [vD'] vR' -> {vL i VD vR; VL' j vU' vR'}

        if n > 0:
            assert n <= self._nc - 1
            assert 0 <= i < self._L[n] - 1
            return np.tensordot(
                self.get_theta1(n, i), self.ttnB[n][i + 1], axes=1
            )  # vL i _vU_ _vD_ [vR],  [vL] j _vU_ _vD_ vR -> {vL i _vU_ _vD_; j _vU_ _vD_ vR}

    def split_truncate_theta(self, theta, n, i, chi_max, eps):
        """
        Split the contracted two-site wave function and truncate the number of singular values.
        :param theta:
        :type theta:
        :param n: Which chain. n=-1 means the backbone.
        In this case (n=-1), i means which bond: 0 -> the first.
        :type n: int
        :param i: which bond on the chain
        :type i: int
        :param chi_max: int, Maximum number of singular values to keep
        :type chi_max: int
        :param eps:
        :type eps: float
        :return: No return. Just updates the tensors in self.ttnS and self.ttnB
        """
        #theta = self.get_theta2(n, i)
        if n == -1:
            # {Down part: vL i VD vR; Up part: VL' j vU' vR'}
            chiL_d, p_d, chiD_d, chiR_d, chiL_u, p_u, chiU_u, chiR_u = theta.shape
            theta = np.reshape(theta, [chiL_d * p_d * chiD_d * chiR_d,
                                       chiL_u * p_u * chiU_u * chiR_u])
            D, S, U = svd(theta, full_matrices=False)
            chivC = min(chi_max, np.sum(S > eps))
            piv = np.argsort(S)[::-1][:chivC]  # keep the largest `chivC` singular values
            D, S, U = D[:, piv], S[piv], U[piv, :]
            S = S / np.linalg.norm(S)
            D = np.reshape(D, [chiL_d, p_d, chiD_d, chiR_d, chivC])
            # D: vL*i*vD*vR*chivC -> vL i vD vR vU=chivC
            D = np.transpose(D, [0, 1, 4, 2, 3])
            # vL i vD vR vU -> vL i vU vD vR
            U = np.reshape(U, [chivC, chiL_u, p_u, chiU_u, chiR_u])
            # B: chivC*vL*j*vU*vR -> vD==chivC vL i vU vR
            U = np.transpose(U, [1, 2, 3, 0, 4])
            # vD vL i vU vR -> vL i vU vD vR
            self.ttnS[i][-1] = S
            D = np.tensordot(
                D, np.diag(S),
                [2, 0]
            )  # vL i [vU] vD vR, [vU'] vU -> vL i vD vR vU
            D = np.transpose(D, [0, 1, 4, 3, 2])
            D = np.tensordot(
                np.diag(self.ttnS[i][self._ebL[i]]) ** (-1), D,
                [1, 0]
            )  # vL [vL'], [vL'] i vU vD vR -> vL i vD vR vU
            self.ttnB[i][self._ebL[i]] = D
            U = np.tensordot(
                U, np.diag(S),
                [2, 0]
            )  # vL i vU [vD] vR, [vD'] vD -> vL i vU vR vD
            U = np.transpose(U, [0, 1, 2, 4, 3])
            # vL i vU vR vD -> vL i vU vD vR
            U = np.tensordot(
                np.diag(self.ttnS[i+1][self._ebL[i+1]]) ** (-1), U,
                [1, 0]
            )  # vL [vL'], [vL'] i vU vD vR -> vL i vU vD vR
            self.ttnB[i+1][self._ebL[i+1]] = U

        else:
            if i == self._vbL[n]:
                chiL_l, p_l, chiR_l, p_r, chiU_r, chiD_r, chiR_r = theta.shape
                theta = np.reshape(theta, [chiL_l * p_l * chiR_l,
                                           p_r * chiU_r * chiD_r * chiR_r])
                A, S, B = svd(theta, full_matrices=False)
                chivC = min(chi_max, np.sum(S > eps))
                piv = np.argsort(S)[::-1][:chivC]  # keep the largest `chivC` singular values
                A, S, B = A[:, piv], S[piv], B[piv, :]
                S = S / np.linalg.norm(S)
                self.ttnS[n][i+1] = S
                A = np.reshape(A, [chiL_l, p_l, chivC])
                # A: vL*i*vR -> vL i vR=chivC
                B = np.reshape(B, [chivC, p_r, chiU_r, chiD_r, chiR_r])
                # B: chivC*j*vU*vD*vR -> vL==chivC j vU vD vR
                A = np.tensordot(np.diag(self.ttnS[n][i-1]**(-1)), A, axes=[1, 0])
                A = np.tensordot(A, np.diag(S), axes=[2,0])
                self.ttnB[n][i] = A
                self.ttnB[n][i+1] = B

            elif i == self._vbL[n] + 1:
                chiL_l, p_l, chiU_l, chiD_l, p_r, chiR_r = theta.shape
                theta = np.reshape(theta, [chiL_l * p_l * chiU_l * chiD_l,
                                           p_r *  chiR_r])
                A, S, B = svd(theta, full_matrices=False)
                chivC = min(chi_max, np.sum(S > eps))
                piv = np.argsort(S)[::-1][:chivC]  # keep the largest `chivC` singular values
                A, S, B = A[:, piv], S[piv], B[piv, :]
                S = S / np.linalg.norm(S)
                self.ttnS[n][i+1] = S
                A = np.reshape(A, [chiL_l, p_l, chiU_l, chiD_l, chivC])
                # A: {vL*i*vU*vD, chivC} -> vL i vU vD vR=chivC
                B = np.reshape(B, [chivC, p_r,  chiR_r])
                # B: {chivC, j*vR} -> vL==chivC j vR
                A = np.tensordot(np.diag(self.ttnS[n][i-1]**(-1)), A, axes=[1, 0])
                # vL [vL'] * [vL] i vU vD vR -> vL i vU vD vR
                A = np.tensordot(A, S, [4, 0])
                # vL i vU vD [vR] * [vR] vR -> vL i vU vD vR
                self.ttnB[n][i] = A
                self.ttnB[n][i+1] = B

            else:
                chiL_l, p_l, p_r, chiR_r = theta.shape
                theta = np.reshape(theta, [chiL_l * p_l,
                                           p_r * chiR_r])
                A, S, B = svd(theta, full_matrices=False)
                chivC = min(chi_max, np.sum(S > eps))
                piv = np.argsort(S)[::-1][:chivC]  # keep the largest `chivC` singular values
                A, S, B = A[:, piv], S[piv], B[piv, :]
                S = S / np.linalg.norm(S)
                self.ttnS[n][i+1] = S
                A = np.reshape(A, [chiL_l, p_l, chivC])
                # A: {vL*i, chivC} -> vL i vR=chivC
                B = np.reshape(B, [chivC, p_r, chiR_r])
                # B: {chivC, j*vR} -> vL==chivC j vR
                A = np.tensordot(np.diag(self.ttnS[n][i - 1] ** (-1)), A, axes=[1, 0])
                # vL [vL'] * [vL] i vR -> vL i vR
                A = np.tensordot(A, S, [4, 0])
                # vL i [vR] * [vR] vR -> vL i vR
                self.ttnB[n][i] = A
                self.ttnB[n][i + 1] = B

    def update_bond(self, n, i, chi_max, eps):
        """
        :param n:
        :type n:
        :param i:
        :type i:
        :param chi_max:
        :type chi_max:
        :param eps:
        :type eps:
        """
        theta = self.get_theta2(n, i)
        if n == -1:
            # {Down part: vL i VD vR; Up part: VL' j vU' vR'}
            Utheta = np.einsum('IJKL,aKcdLfgh->aIcdJfgh', self.ttnH[i][-1], theta)
            # {i j [i*] [j*]} * {vL [i] vU vD, [j] vU vD vR}
            # {i j   k   l}     {a   b  c  d ,  e  f  g  h}
            self.split_truncate_theta(Utheta, n, i, chi_max, eps)
        elif n >= 0:
            if i == self._vbL[n]:
                Utheta = np.einsum('IJKL,aKLfgh->aIJfgh', self.ttnH[n][i], theta)
                # {i j [i*] [j*]} * {vL [i]  [j] vU vD vR}
                # {i j  k   l}      {a   b   e   f  g  h}
                self.split_truncate_theta(Utheta, n, i, chi_max, eps)

            elif i == self._vbL[n] + 1:
                Utheta = np.einsum('IJKL,aKcdLh->aIcdJh', self.ttnH[n][i], theta)
                # {i j [i*] [j*]} * {vL [i] vU vD,  [j]  vR}
                # {i j  k   l}      {a   b  c  d ,   e   h}
                self.split_truncate_theta(Utheta, n, i, chi_max, eps)

            else:
                Utheta = np.einsum('IJKL,aKLh->aIJh', self.ttnH[n][i], theta)
                # {i j [i*] [j*]} * {vL [i], [j] vR}
                # {I J  K   L}      {a   b,   e   h}
                self.split_truncate_theta(Utheta, n, i, chi_max, eps)

def init_ttn(nc, L, d1, d2):
    """
    Initialize the SimpleTTN class.
    Fill all the relevant lists, including ttnS, ttnB, ttnH.
    :param nc:
    :type nc:
    :param L:
    :type L:
    :param d1:
    :type d1:
    :param d2:
    :type d2:
    :return:
    :rtype:
    """
    eb = np.zeros([1, d1, 1], np.float)  # vL i vR
    eb[0, 0, 0] = 1.
    ebs = [eb.copy() for i in range(L)]
    ebss = [ebs.copy() for i in range(nc)]
    #
    ev = np.zeros([1, d2, 1, 1, 1], np.float)  # vL i vU vD vR
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
    ev_ss = [ev_s.copy() for i in range(2)]  # we have two sites here, e site and v site.
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
    ttn = init_ttn(nc=4, L=3, d1=5, d2=6)
    ttn.get_theta2(-1, 1)
