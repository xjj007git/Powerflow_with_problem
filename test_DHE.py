import HELMpy
import time
import numpy as np
import pandapower.networks as pn
from scipy.stats import truncnorm
from pandapower.run import runpp, set_user_pf_options
import pandapower as pp

if __name__ == "__main__":
    net_HE = HELMpy.HELM(pn.case118())
    Num_Samples = 1000
    baseMVA = net_HE.net._ppc["baseMVA"]

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

    a = (0.5 - 1.0) / 0.05   # = -2.0
    b = (1.2 - 1.0) / 0.05   # = 2.0
    load_scales_S_pq = truncnorm.rvs(a, b, loc=1.0, scale=0.05, size=(Num_Samples, len(S_pq)))
    load_scales_P_pv = truncnorm.rvs(a, b, loc=1.0, scale=0.05, size=(Num_Samples, len(P_pv)))

    times_DHE  = []
    terms_DHE  = []

    for i in range(Num_Samples):
        new_S_pq = S_pq * load_scales_S_pq[i]
        new_P_pv = P_pv * load_scales_P_pv[i]
        new_Vsp  = Vsp

        start = time.perf_counter()
        V_DHE,S_DHE,terms_DHE_ = net_HE.run_DHE(new_S_pq, new_P_pv, new_Vsp)
        end = time.perf_counter() 
        elapsed_ms = (end - start) * 1000
        times_DHE.append(elapsed_ms)
        terms_DHE.append(terms_DHE_)

    print(f"DHE平均运行时间:{np.mean(times_DHE):.2f}ms")
    print(f"DHE平均迭代次数:{np.mean(terms_DHE):.2f}")
    ###DHE收敛就更加容易得多，比起HE它更新了Y矩阵，虽然他们都很看重初始值germ，修改DHE使其能够收敛方法一般推荐
        #修改HE得迭代次数，使得初始解更加优秀，更容易让DHE收敛
        #修改DHE自身得迭代次数，这一点通常都非常有效
