import HELMpy
import time
import numpy as np
import pandapower.networks as pn
from scipy.stats import truncnorm
from pandapower.run import runpp, set_user_pf_options
import pandapower as pp

if __name__ == "__main__":
    # %%
    net = pn.case118()
    runpp(net)
    Num_Samples = 1000
    baseMVA = net._ppc["baseMVA"]
    pv = net._ppc["internal"]["pv"]
    pq = net._ppc["internal"]["pq"]
    nb = net.bus.shape[0]
    ref= net._ppc["internal"]["ref"]

    S_total = np.zeros(nb, dtype=np.complex128)
    S_total[net.gen.bus] = net.gen.p_mw
    S_total[net.load.bus] = S_total[net.load.bus] - net.load.p_mw - 1j * net.load.q_mvar
    if len(net.sgen.bus) > 0:
        S_total[net.sgen.bus] = S_total[net.sgen.bus] + net.sgen.p_mw
    S_total = S_total/net._ppc["baseMVA"]

    P_pv = np.real(S_total[pv])
    S_pq = S_total[pq]
    Vsp  = np.ones(nb)
    Vsp[ref] = net.ext_grid.vm_pu
    Vsp[net.gen.bus] = net.gen.vm_pu

    a = (0.9 - 1.0) / 0.05   # = -2.0
    b = (1.1 - 1.0) / 0.05   # = 2.0
    load_scales_S_pq = truncnorm.rvs(a, b, loc=1.0, scale=0.05, size=(Num_Samples, len(S_pq)))
    load_scales_P_pv = truncnorm.rvs(a, b, loc=1.0, scale=0.05, size=(Num_Samples, len(P_pv)))

    times_NR   = []
    terms_NR   = []

    for i in range(Num_Samples):
        new_S_pq = S_pq * load_scales_S_pq[i]
        new_P_pv = P_pv * load_scales_P_pv[i]
        new_Vsp  = Vsp

        start = time.perf_counter()
        #修改传入runpp的net网络参数
        # net_HE.net.gen.p_mw=P_pv*baseMVA
        #获取纯负载节点的母线索引，然后获取纯负载在load中的索引
        load_only_buses = np.setdiff1d(net.load.bus, net.gen.bus)
        keys = [key for key, val in net.load.bus.items() if val in load_only_buses]
        #还原一个new_S_total，不影响原本的S_total，不影响DHE计算，同时new_S_total将作为选取纯负载节点功率的容器，但是仅仅更新new_S_total比较麻烦，这里想到的办法是连同所有pq节点全部更新了
        new_S_total = S_total.copy()
        new_S_total[pq] = new_S_pq 

        net.load.loc[keys,'p_mw'] = abs(np.real(new_S_total[load_only_buses])*baseMVA)
        net.load.loc[keys,'q_mvar'] = np.imag(new_S_total[load_only_buses])*baseMVA

        runpp(net, max_iteration=100, tolerance_mva=1e-4,)
        end = time.perf_counter() 
        elapsed_ms = (end - start) * 1000
        times_NR.append(elapsed_ms)
        terms_NR.append(net._ppc["iterations"])

    print(f"NR平均运行时间：{np.mean(times_NR):.2f}ms")
    print(f"NR平均迭代次数：{np.mean(terms_NR):.2f}")
    # pp.diagnostic(net) 