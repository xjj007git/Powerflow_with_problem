import numpy as np
import pandapower as pp
from scipy.sparse import csr_matrix
import pandapower.networks as pn
import scipy
import time
    
class HELM(object):
    
    def __init__(self,net=pn.case5()):

        self.K1 = 20
        self.K2 = 100
        
        self.net=net
        pp.runpp(net)
        self.Ybus    = net._ppc["internal"]["Ybus"]
        self.ref     = net._ppc["internal"]["ref"]#平衡节点
        self.pv      = net._ppc["internal"]["pv"]
        self.pq      = net._ppc["internal"]["pq"]
        self.nb      = net.bus.shape[0]
        self.npv     = self.pv.shape[0]
        self.Ysh     = np.squeeze(np.array(np.sum(self.Ybus,1)))#自导纳，和和其他节点的导纳消去了

        nb, ref = self.nb, self.ref
        npv, pv = self.npv, self.pv
        Ytr            =  self.Ybus - np.diag(self.Ysh)#串联导纳
        Ytr[ ref,:]    = 0
        Ytr[ ref, ref] = 1
        Gtr          = np.real(Ytr)
        Btr          = np.imag(Ytr)
        Y_lhs        = csr_matrix(np.vstack([np.hstack([Gtr, -Btr]), np.hstack([Btr, Gtr])]))
        nb_pv_mat    = csr_matrix( (np.ones(npv), (pv, np.arange(npv))), shape= (nb, npv))
        
        Y_LHS      = np.vstack([np.hstack([Y_lhs.toarray(), np.vstack([np.zeros((nb,npv)), nb_pv_mat.toarray()])]),
                                np.hstack([nb_pv_mat.toarray().T, np.zeros((npv,nb+npv))])])
        self.lu0, self.piv0  = scipy.linalg.lu_factor(Y_LHS)

        S_total = np.zeros(nb, dtype=np.complex128)
        S_total[net.gen.bus] = net.gen.p_mw
        S_total[net.load.bus] = S_total[net.load.bus] - net.load.p_mw - 1j * net.load.q_mvar
        if len(net.sgen.bus) > 0:
            S_total[net.sgen.bus] = S_total[net.sgen.bus] + net.sgen.p_mw   #每个节点注入功率

        S_total = S_total / net._ppc["baseMVA"]

        P_pv = np.real(S_total[pv])
        S_pq = S_total[self.pq]
        Vsp = np.ones(nb)
        Vsp[ref] = net.ext_grid.vm_pu
        Vsp[net.gen.bus] = net.gen.vm_pu

        V_HE, S_HE ,terms= self.run_HE(S_pq, P_pv, Vsp)
        self.V0, self.S0 = V_HE, S_HE
        self.update_Y_LHS_dynamic()

    def pade_approximation(self, V_coeffs):
        """
        对幂级数系数应用帕德近似（对角/次对角），返回 alpha=1 处的电压。
        V_coeffs: shape [n_buses, N], N = k+1 (k>=20 时 N>=21) N代表列数
        """
        n_buses, N = V_coeffs.shape
        
        #根据 N 自动选取 L 和 M (满足 L+M+1 = N),这里采用对角帕德
        if N % 2 == 1:   # N=21,23,25... 取 L=M
            L = M = (N - 1) // 2
        else:            # N=20,22,24... 取 L=M-1 (次对角)
            L = N // 2 - 1
            M = N // 2
        
        #初始化分母 Q: q0=1, 求 q1...qM
        q = np.zeros((n_buses, M + 1), dtype=np.complex128)
        q[:, 0] = 1.0 + 0.0j
        
        #构造方程组 A * q[1:] = b
        A = np.zeros((n_buses, M, M), dtype=np.complex128)
        b = np.zeros((n_buses, M), dtype=np.complex128)
        
        for i in range(M):
            b[:, i] = -V_coeffs[:, L + 1 + i]
            for j in range(1, M + 1):
                A[:, i, j - 1] = V_coeffs[:, L + 1 + i - j]
        
        #逐母线求解
        for bus in range(n_buses):
            try:
                q[bus, 1:] = np.linalg.solve(A[bus, :, :], b[bus, :])
            except np.linalg.LinAlgError:
                print('出现奇异矩阵！solve求解不了')
                q[bus, 1:], _, _, _ = np.linalg.lstsq(A[bus, :, :], b[bus, :], rcond=None) #最小二乘法
        
        #计算分子P
        P = np.zeros((n_buses, L + 1), dtype=np.complex128)
        for n in range(L + 1):
            for j in range(0, min(M, n) + 1):
                P[:, n] += q[:, j] * V_coeffs[:, n - j]
        
        #在α=1处求值
        V_pade = np.sum(P, axis=1) / np.sum(q, axis=1)
        return V_pade
    
    def update_Y_LHS_dynamic(self):

        nb, ref = self.nb, self.ref
        npv, pv = self.npv, self.pv

        W = 1 / self.V0
        GermUpdate = np.diag(self.S0 * W / self.V0)

        Y_LHS = np.vstack(
            [np.hstack([np.real(self.Ybus) + np.real(GermUpdate), -np.imag(self.Ybus) - np.imag(GermUpdate),
                        csr_matrix((np.imag(W[pv]), (pv, np.arange(npv))), shape=(nb, npv)).toarray()
                        ]),
             np.hstack([np.imag(self.Ybus) - np.imag(GermUpdate), np.real(self.Ybus) - np.real(GermUpdate),
                        csr_matrix((np.real(W[pv]), (pv, np.arange(npv))), shape=(nb, npv)).toarray()
                        ]),
             np.hstack([csr_matrix((np.real(self.V0[pv]), (np.arange(npv), pv)), shape=(npv, nb)).toarray(),
                        csr_matrix((np.imag(self.V0[pv]), (np.arange(npv), pv)), shape=(npv, nb)).toarray(),
                        np.zeros((npv, npv))
                        ])
             ])

        Y_LHS[ref, :], Y_LHS[ref + nb, :] = 0, 0
        Y_LHS[ref, ref], Y_LHS[ref + nb, ref + nb] = 1, 1
        self.lu, self.piv = scipy.linalg.lu_factor(Y_LHS)

    def run_HE(self, S_pq, P_pv, Vsp):

        nb, ref     = self.nb, self.ref
        npv, pv, pq = self.npv, self.pv, self.pq
        k = 0
        RHS    = np.zeros(nb, dtype=np.complex128)

        # V      = np.ones((nb,  2000), dtype=np.complex128)
        # W      = np.ones((nb,  2000), dtype=np.complex128)
        # Q      = np.zeros((npv, 2000), dtype=np.complex128)

        V      = np.ones((nb,  100), dtype=np.complex128)
        W      = np.ones((nb,  100), dtype=np.complex128)
        Q      = np.zeros((npv, 100), dtype=np.complex128)

        while True:
            k = k + 1 
            if (k==1):
                RHS[pq]  = np.conj(S_pq) *W[pq,0]
                RHS[pv]  = P_pv*W[pv,0]
                RHS      = RHS - self.Ysh *V[:,0]
                RHS[ref] = Vsp[ref] - V[ref,0]
                RHS_Vr   = (np.abs(Vsp[pv])**2 - V[pv,0]**2) /2
                 
            else:
                RHS[pq]  = np.conj( S_pq *W[pq,k-1] )
                RHS[pv]  = np.conj( P_pv *W[pv,k-1] ) - 1j*self.f_Q(Q[:,0:k],W[pv,0:k])
                RHS      = RHS - self.Ysh * V[:,k-1]
                RHS[ref] = 0
                RHS_Vr   = self.f_V(V[pv,0:k],V[pv,0:k])
            
            RHS_all  = np.concatenate([np.real(RHS), np.imag(RHS), RHS_Vr])
            LHS = scipy.linalg.lu_solve((self.lu0, self.piv0), RHS_all)

            V[:,k]   = LHS[:nb] + 1j*LHS[nb:nb*2]
            Q[:,k]   = LHS[nb*2:]

            VV1 = np.sum(V[:,0:k+1],axis=1)

            S_cal = VV1*np.conj(self.Ybus@VV1)
            del_S  = np.absolute(np.concatenate([S_pq-S_cal[pq], P_pv-np.real(S_cal[pv])]) ).max()
            del_V  = np.max(np.abs(np.abs(VV1[pv])-np.abs(Vsp[pv])))

            if del_S <= 1e-4 and del_V <= 1e-4:
                break
            else:
                W[:,k]   = self.fcon(V[:,0:k+1],W[:,0:k])

            # if del_S <= 1e-1 and del_V <= 1e-1:
            #     break
            # else:
            #     W[:,k]   = self.fcon(V[:,0:k+1],W[:,0:k])

        return VV1, S_cal, k

    def run_DHE(self, S_pq, P_pv, Vsp):

        S0, V0 = self.S0, self.V0

        nb, ref = self.nb, self.ref
        npv, pv, pq = self.npv, self.pv, self.pq
        k = 0
        RHS = np.zeros(nb, dtype=np.complex128)

        V = np.ones((nb, 100), dtype=np.complex128)
        W = np.ones((nb, 100), dtype=np.complex128)
        Q = np.zeros((npv, 100), dtype=np.complex128)

        V[:, 0] = V0 #这里已经计算了V[0]了，也就是k==0
        Q[:, 0] = np.imag(S0[pv])
        W[:, 0] = 1 / V0
        P_pv0 = np.real(S0[pv])
        S_pq0 = S0[pq]

        while k <= (self.K2 -2):
            k = k + 1
            #计算V的各个几点的系数
            if (k == 1):
                RHS[pq] = np.conj((S_pq - S_pq0) * W[pq, k - 1])
                RHS[pv] = np.conj((P_pv - P_pv0) * W[pv, k - 1])
                RHS[ref] = Vsp[ref] - V0[ref]
                RHS_Vr = (np.abs(Vsp[pv]) ** 2 - np.abs(V0[pv]) ** 2) / 2

            else:
                RHS[pq] = np.conj(S_pq0 * self.fcon_1(V[pq, 0:k], W[pq, 0:k])) + np.conj(
                    (S_pq - S_pq0) * W[pq, k - 1])
                RHS[pv] = np.conj((P_pv0 + 1j * Q[:, 0]) * self.fcon_1(V[pv, 0:k], W[pv, 0:k])) + np.conj(
                    (P_pv - P_pv0) * W[pv, k - 1]) - 1j * self.f_Q(Q[:, 0:k], W[pv, 0:k])
                RHS[ref] = 0
                RHS_Vr = self.f_V(V[pv, 0:k], V[pv, 0:k])

            RHS_all = np.concatenate([np.real(RHS), np.imag(RHS), RHS_Vr])
            LHS = scipy.linalg.lu_solve((self.lu, self.piv), RHS_all)

            V[:, k] = LHS[:nb] + 1j * LHS[nb:nb * 2]
            Q[:, k] = LHS[nb * 2:]
            
            #现在就是根据k的迭代次数用不同方法求各个节点电压了
            # if k >= self.K1:
            #     # print('进入pade近似')
            #     VV1 = self.pade_approximation(V)
            # else:
            #     VV1 = np.sum(V[:, 0:k + 1], axis=1)

            # #测试看一下pade近似和求和级数的区别
            # print(f'这是pade的结果：{self.pade_approximation(V)}')
            # print(f'这是求和的结果：{np.sum(V[:, 0:k + 1], axis=1)}')

            VV1 = np.sum(V[:, 0:k + 1], axis=1)

            # Check convergence
            S_cal = VV1 * np.conj(self.Ybus @ VV1)
            del_S = np.absolute(np.concatenate([S_pq - S_cal[pq], P_pv - np.real(S_cal[pv])])).max()
            # del_V = np.max(np.abs(np.abs(VV1[pv]) - np.abs(Vsp[pv])))
            #PV节点转化后,可能没有PV节点了
            del_V = np.max(np.abs(np.abs(VV1[pv]) - np.abs(Vsp[pv]))) if len(pv) > 0 else 0

            if del_S <= 1e-4 and del_V <= 1e-4:
                break
            else:
                W[:, k] = self.fcon(V[:, 0:k + 1], W[:, 0:k])
        
        return VV1, S_cal, k

    def fcon(self, V_,W_):
        return -np.sum(V_[:,1:]     *np.fliplr(W_), axis=1) / V_[:,0]
    
    def f_Q(self, Q_,W_):
        return  np.sum(Q_[:,1:]     *np.conj(np.fliplr(W_[:,1:])), axis= 1)
    
    def f_V(self, V1_,V2_):
        return -np.sum(V1_[:,1:]    *np.conj(np.fliplr(V2_[:,1:])), axis=1) /2

    def fcon_1(self, V_,W_):
        return -np.sum(V_[:,1:]     *np.fliplr(W_[:,1:]), axis=1) /V_[:,0]

class quick_HELM(object):
    
    def __init__(self,net=pn.case5()):

        self.net=net
        pp.runpp(net)
        self.Ybus    = net._ppc["internal"]["Ybus"]
        self.ref     = net._ppc["internal"]["ref"]#平衡节点
        self.pv      = net._ppc["internal"]["pv"]
        self.pq      = net._ppc["internal"]["pq"]
        self.nb      = net.bus.shape[0]
        self.npv     = self.pv.shape[0]
        self.Ysh     = np.squeeze(np.array(np.sum(self.Ybus,1)))#自导纳，和和其他节点的导纳消去了

        nb, ref = self.nb, self.ref
        npv, pv = self.npv, self.pv
        Ytr            =  self.Ybus - np.diag(self.Ysh)#串联导纳
        Ytr[ ref,:]    = 0
        Ytr[ ref, ref] = 1
        Gtr          = np.real(Ytr)
        Btr          = np.imag(Ytr)
        Y_lhs        = csr_matrix(np.vstack([np.hstack([Gtr, -Btr]), np.hstack([Btr, Gtr])]))
        nb_pv_mat    = csr_matrix( (np.ones(npv), (pv, np.arange(npv))), shape= (nb, npv))
        
        Y_LHS      = np.vstack([np.hstack([Y_lhs.toarray(), np.vstack([np.zeros((nb,npv)), nb_pv_mat.toarray()])]),
                                np.hstack([nb_pv_mat.toarray().T, np.zeros((npv,nb+npv))])])
        self.lu0, self.piv0  = scipy.linalg.lu_factor(Y_LHS)

        S_total = np.zeros(nb, dtype=np.complex128)
        S_total[net.gen.bus] = net.gen.p_mw
        S_total[net.load.bus] = S_total[net.load.bus] - net.load.p_mw - 1j * net.load.q_mvar
        if len(net.sgen.bus) > 0:
            S_total[net.sgen.bus] = S_total[net.sgen.bus] + net.sgen.p_mw   #每个节点注入功率

        S_total = S_total / net._ppc["baseMVA"]

        P_pv = np.real(S_total[pv])
        S_pq = S_total[self.pq]
        Vsp = np.ones(nb)
        Vsp[ref] = net.ext_grid.vm_pu
        Vsp[net.gen.bus] = net.gen.vm_pu

        V_HE, S_HE ,terms= self.run_HE(S_pq, P_pv, Vsp)
        self.V0, self.S0 = V_HE, S_HE
        self.update_Y_LHS_dynamic()

    def update_Y_LHS_dynamic(self):

        nb, ref = self.nb, self.ref
        npv, pv = self.npv, self.pv

        W = 1 / self.V0
        GermUpdate = np.diag(self.S0 * W / self.V0)

        Y_LHS = np.vstack(
            [np.hstack([np.real(self.Ybus) + np.real(GermUpdate), -np.imag(self.Ybus) - np.imag(GermUpdate),
                        csr_matrix((np.imag(W[pv]), (pv, np.arange(npv))), shape=(nb, npv)).toarray()
                        ]),
             np.hstack([np.imag(self.Ybus) - np.imag(GermUpdate), np.real(self.Ybus) - np.real(GermUpdate),
                        csr_matrix((np.real(W[pv]), (pv, np.arange(npv))), shape=(nb, npv)).toarray()
                        ]),
             np.hstack([csr_matrix((np.real(self.V0[pv]), (np.arange(npv), pv)), shape=(npv, nb)).toarray(),
                        csr_matrix((np.imag(self.V0[pv]), (np.arange(npv), pv)), shape=(npv, nb)).toarray(),
                        np.zeros((npv, npv))
                        ])
             ])

        Y_LHS[ref, :], Y_LHS[ref + nb, :] = 0, 0
        Y_LHS[ref, ref], Y_LHS[ref + nb, ref + nb] = 1, 1
        self.lu, self.piv = scipy.linalg.lu_factor(Y_LHS)

    def run_HE(self, S_pq, P_pv, Vsp):

        nb, ref     = self.nb, self.ref
        npv, pv, pq = self.npv, self.pv, self.pq
        k = 0
        RHS    = np.zeros(nb, dtype=np.complex128)

        # V      = np.ones((nb,  2000), dtype=np.complex128)
        # W      = np.ones((nb,  2000), dtype=np.complex128)
        # Q      = np.zeros((npv, 2000), dtype=np.complex128)

        V      = np.ones((nb,  100), dtype=np.complex128)
        W      = np.ones((nb,  100), dtype=np.complex128)
        Q      = np.zeros((npv, 100), dtype=np.complex128)

        while True:
            k = k + 1 
            if (k==1):
                RHS[pq]  = np.conj(S_pq) *W[pq,0]
                RHS[pv]  = P_pv*W[pv,0]
                RHS      = RHS - self.Ysh *V[:,0]
                RHS[ref] = Vsp[ref] - V[ref,0]
                RHS_Vr   = (np.abs(Vsp[pv])**2 - V[pv,0]**2) /2
                 
            else:
                RHS[pq]  = np.conj( S_pq *W[pq,k-1] )
                RHS[pv]  = np.conj( P_pv *W[pv,k-1] ) - 1j*self.f_Q(Q[:,0:k],W[pv,0:k])
                RHS      = RHS - self.Ysh * V[:,k-1]
                RHS[ref] = 0
                RHS_Vr   = self.f_V(V[pv,0:k],V[pv,0:k])
            
            RHS_all  = np.concatenate([np.real(RHS), np.imag(RHS), RHS_Vr])
            LHS = scipy.linalg.lu_solve((self.lu0, self.piv0), RHS_all)

            V[:,k]   = LHS[:nb] + 1j*LHS[nb:nb*2]
            Q[:,k]   = LHS[nb*2:]

            VV1 = np.sum(V[:,0:k+1],axis=1)

            S_cal = VV1*np.conj(self.Ybus@VV1)
            del_S  = np.absolute(np.concatenate([S_pq-S_cal[pq], P_pv-np.real(S_cal[pv])]) ).max()
            del_V  = np.max(np.abs(np.abs(VV1[pv])-np.abs(Vsp[pv])))

            # if del_S <= 1e-4 and del_V <= 1e-4:
            #     break
            # else:
            #     W[:,k]   = self.fcon(V[:,0:k+1],W[:,0:k])

            if del_S <= 1e-2 and del_V <= 1e-2:
                break
            else:
                W[:,k]   = self.fcon(V[:,0:k+1],W[:,0:k])

        return VV1, S_cal, k

    def run_DHE(self, S_pq, P_pv, Vsp, pv, pq):

        iter_max = 9
        S0, V0 = self.S0, self.V0

        nb, ref = self.nb, self.ref
        npv, pv, pq = self.npv, self.pv, self.pq
        k = 0
        RHS = np.zeros(nb, dtype=np.complex128)

        V = np.ones((nb, 100), dtype=np.complex128)
        W = np.ones((nb, 100), dtype=np.complex128)
        Q = np.zeros((npv, 100), dtype=np.complex128)

        # V = np.ones((nb, 1500), dtype=np.complex128)
        # W = np.ones((nb, 1500), dtype=np.complex128)
        # Q = np.zeros((npv, 1500), dtype=np.complex128)

        V[:, 0] = V0
        Q[:, 0] = np.imag(S0[pv])
        W[:, 0] = 1 / V0
        P_pv0 = np.real(S0[pv])
        S_pq0 = S0[pq]

        while k < iter_max: #k从0到8
            k = k + 1       #k从1到9,这里直接代表项数，加上之前有的第0项一共10项
            if (k == 1):
                RHS[pq] = np.conj((S_pq - S_pq0) * W[pq, k - 1])
                RHS[pv] = np.conj((P_pv - P_pv0) * W[pv, k - 1])
                RHS[ref] = Vsp[ref] - V0[ref]
                RHS_Vr = (np.abs(Vsp[pv]) ** 2 - np.abs(V0[pv]) ** 2) / 2

            else:
                RHS[pq] = np.conj(S_pq0 * self.fcon_1(V[pq, 0:k], W[pq, 0:k])) + np.conj(
                    (S_pq - S_pq0) * W[pq, k - 1])
                RHS[pv] = np.conj((P_pv0 + 1j * Q[:, 0]) * self.fcon_1(V[pv, 0:k], W[pv, 0:k])) + np.conj(
                    (P_pv - P_pv0) * W[pv, k - 1]) - 1j * self.f_Q(Q[:, 0:k], W[pv, 0:k])
                RHS[ref] = 0
                RHS_Vr = self.f_V(V[pv, 0:k], V[pv, 0:k])

            RHS_all = np.concatenate([np.real(RHS), np.imag(RHS), RHS_Vr])
            LHS = scipy.linalg.lu_solve((self.lu, self.piv), RHS_all)

            V[:, k] = LHS[:nb] + 1j * LHS[nb:nb * 2]
            Q[:, k] = LHS[nb * 2:]

            VV1 = np.sum(V[:, 0:k + 1], axis=1) #没有K1的使用，也就是说没有帕德近似

            # Check convergence
            S_cal = VV1 * np.conj(self.Ybus @ VV1)
            del_S = np.absolute(np.concatenate([S_pq - S_cal[pq], P_pv - np.real(S_cal[pv])])).max()
            del_V = np.max(np.abs(np.abs(VV1[pv]) - np.abs(Vsp[pv])))

            if del_S <= 1e-4 and del_V <= 1e-4:
                break
            else:
                W[:, k] = self.fcon(V[:, 0:k + 1], W[:, 0:k])

        return VV1, S_cal, k

    def fcon(self, V_,W_):
        return -np.sum(V_[:,1:]     *np.fliplr(W_), axis=1) / V_[:,0]
    
    def f_Q(self, Q_,W_):
        return  np.sum(Q_[:,1:]     *np.conj(np.fliplr(W_[:,1:])), axis= 1)
    
    def f_V(self, V1_,V2_):
        return -np.sum(V1_[:,1:]    *np.conj(np.fliplr(V2_[:,1:])), axis=1) /2

    def fcon_1(self, V_,W_):
        return -np.sum(V_[:,1:]     *np.fliplr(W_[:,1:]), axis=1) /V_[:,0]

if __name__ == '__main__':
    net_HE = HELM(pn.case300())

    S_total = np.zeros(net_HE.nb, dtype=np.complex128)
    S_total[net_HE.net.gen.bus] = net_HE.net.gen.p_mw
    S_total[net_HE.net.load.bus] = S_total[net_HE.net.load.bus] - net_HE.net.load.p_mw - 1j * net_HE.net.load.q_mvar
    if len(net_HE.net.sgen.bus) > 0:
        S_total[net_HE.net.sgen.bus] = S_total[net_HE.net.sgen.bus] + net_HE.net.sgen.p_mw
    S_total = S_total/net_HE.net._ppc["baseMVA"]

    P_pv = np.real(S_total[net_HE.pv])
    S_pq = S_total[net_HE.pq]
    Vsp  = np.ones(net_HE.nb)
    Vsp[net_HE.ref] = net_HE.net.ext_grid.vm_pu
    Vsp[net_HE.net.gen.bus] = net_HE.net.gen.vm_pu

    # V_HE,S_HE = net_HE.run_DHE(S_pq,P_pv, Vsp)

    start = time.perf_counter()
    V_HE,S_HE,terms = net_HE.run_DHE(S_pq,P_pv, Vsp)
    end = time.perf_counter() 
    elapsed_ms = (end - start) * 1000 
    print(f"运行时间：{elapsed_ms:.2f}ms")
    print(f"迭代次数：{terms}")



