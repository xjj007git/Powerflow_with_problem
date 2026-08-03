import HELMpy
import my_HELMpy
from custom_envs import PowerSystemEnv
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit
import matlab.engine
import numpy as np
import copy
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def check_Q_limit(env, state):
    _, info_dcv = env.test_calculate_dcv(state)
    if len(info_dcv['over_limit_indices']) == 0:
        flag = False    #越界索引长度为0,不需要PQ转换
    else:
        flag = True
    return flag, info_dcv

def Translate_PV_to_PQ_HE(env, action, info_dcv, his_over_limit_indices):
    net_HE = copy.deepcopy(env.net_HE)  #每次进PQ修正都会重新从env中提取一个新的HE求解器
    # net_HE = env.net_HE     #这样赋值会改变env的net_HE里面的值

    bus_types = np.zeros(net_HE.nb, dtype=int)
    bus_types[net_HE.ref] = 3
    bus_types[net_HE.pv] = 2    #不包括松弛节点
    bus_types[net_HE.pq] = 1

    state = env.current_state.copy()

    #这里应该找出Q越界的节点
    over_limit_indices = info_dcv['over_limit_indices']
    over_limit_indices_include_slack = info_dcv['over_limit_indices_include_slack']     #包含松弛节点

    #现在将over_limit整合his_over_limit防止循环
    over_limit_indices = list(set(over_limit_indices) | set(his_over_limit_indices))  

    #所有gen边界
    q_non_slack_min = info_dcv['q_non_slack_min']
    q_non_slack_max = info_dcv['q_non_slack_max']
    q_slack_min = info_dcv['q_slack_min']
    q_slack_max = info_dcv['q_slack_max']
    #整合成整个gen的Q边界,方便索引取值
    q_all_min = np.concatenate([[q_slack_min], q_non_slack_min])
    q_all_max = np.concatenate([[q_slack_max], q_non_slack_max])
    #所有gen真实数值
    q_slack = info_dcv['q_slack']
    q_non_slack = info_dcv['q_non_slack']
    q_all = np.concatenate([[q_slack], q_non_slack])

    # in_boundary = info_dcv['in_boundary']
    #非松弛节点其他电机的全集索引
    non_slack_all_indices = list(range(env.grid["num_non_slack"]))
    in_boundary = [x for x in non_slack_all_indices if x not in over_limit_indices]

    #获取发电机在母线中的索引
    gen_buses = env.grid['gen_buses']
    non_slack_buses = env.grid['gen_buses'][env.grid['non_slack_indices']]

    over_limit = [] #存储着越界电机的母线索引,以及该母线应该更改成的Q边界
    # print(f"over_limit_indices的长度是:{len(over_limit_indices)}")
    # print(f"over_limit_indices是:{over_limit_indices}")
    for i in over_limit_indices:
        bus = non_slack_buses[i]  #这是越界机组母线索引
        bus_types[bus] = 1  #这个节点修改为pq节点
        if q_non_slack[i] < q_non_slack_min[i]:
            over_limit.append((bus, q_non_slack[i], q_non_slack_min[i], 'min'))
        elif q_non_slack[i] > q_non_slack_max[i]:
            over_limit.append((bus, q_non_slack[i], q_non_slack_max[i], 'max'))
        #针对历史数据,防止越限机组反复循环
        else:
            over_limit.append((bus, q_non_slack[i], q_non_slack_max[i], 'his'))     #直接将无功拉满
    # print(f"over_limit行数是:{len(over_limit)}")
    # print(f"over_limit:{over_limit}")
    # print(f"in_boundary是:{in_boundary}")

    Vsp = np.ones(net_HE.nb)
    S_total = np.zeros(net_HE.nb, dtype=np.complex128)

    # 解析动作
    p_set = action[:env.grid['num_non_slack']]
    v_set = action[env.grid['num_non_slack']:env.grid['num_non_slack']+env.grid['num_gen']]
    load_shed = action[env.grid['num_non_slack']+env.grid['num_gen']:env.grid['num_non_slack']+env.grid['num_gen']+env.grid['num_loads']]
    wind_curt = action[-env.grid['num_wind']:]

    control = {
        'p_set': p_set,      # 后面要从action里面取出未越界节点,其对应的p需要由action获取
        'v_set': v_set,          # v_set应该包含slack和未越界节点
        'load_shed': load_shed,       # 负荷削减比例
        'wind_curt': wind_curt        # 弃风比例
    }
    #取出负载的真实p,q
    actual_load_p = env.re_HE_load_p
    actual_load_q = env.re_HE_load_q

    pq_indices = np.where(bus_types == 1)[0]
    pv_indices = np.where(bus_types == 2)[0]
    ref_indices = np.where(bus_types == 3)[0]   #松弛节点不会改变

    if len(in_boundary) > 0:
        ones_array = np.ones(len(in_boundary)).astype(int)
        in_boundary_temp = in_boundary + ones_array
        in_boundary_plus_slack = np.concatenate([[0], in_boundary_temp])   #加上松弛节点
        Vsp[gen_buses[in_boundary_plus_slack]] = (control['v_set'])[in_boundary_plus_slack]
    if len(in_boundary) == 0:
        pass

    #修改P_pv
    S_total[net_HE.net.gen.bus[in_boundary]] = (control['p_set'])[in_boundary]

    #修改S_pq
    S_total[net_HE.net.load.bus] = S_total[net_HE.net.load.bus] - actual_load_p - 1j * actual_load_q
    if len(net_HE.net.sgen.bus) > 0:
        S_total[net_HE.net.sgen.bus] = S_total[net_HE.net.sgen.bus] + net_HE.net.sgen.p_mw

    #这里已经除去了松弛节点
    q_list = [item[1] for item in over_limit]
    type_list = [item[3] for item in over_limit]
    #下面这两项重要
    boundary_list = [item[2] for item in over_limit]
    bus_list = [item[0] for item in over_limit]
    p_PQ = [x for x in p_set if x not in p_set[in_boundary]]    #获取了p_set中其他越限节点的p_set,传给PQ节点的P
    # print(f"bus_list的长度:{len(bus_list)}")
    # print(f"p_PQ的长度:{len(p_PQ)}")
    # print(f"boundary_list的长度:{len(boundary_list)}")

    boundary_list = np.array(boundary_list)
    p_PQ = np.array(p_PQ)

    #接下来把新的PQ节点数据加进S_pq
    S_total[bus_list] = S_total[bus_list] + p_PQ + 1j * boundary_list
    S_total = S_total/net_HE.net._ppc["baseMVA"]

    #提取数据
    P_pv = np.real(S_total[pv_indices])
    S_pq = S_total[pq_indices]

    net_HE.pv = pv_indices
    net_HE.pq = pq_indices
    net_HE.npv = net_HE.pv.shape[0]

    net_HE.update_Y_LHS_dynamic()

    V_HE,S_HE,terms = net_HE.run_DHE(S_pq, P_pv, Vsp)

    S_HE_mw = S_HE * net_HE.net._ppc["baseMVA"]

    load_on_gen_buses = np.intersect1d(env.grid['load_buses'], env.grid['gen_buses'])

    P_load_on_gen = np.zeros(env.grid['num_gen'])
    Q_load_on_gen = np.zeros(env.grid['num_gen'])

    for bus in load_on_gen_buses:
        gen_idx = np.where(env.grid['gen_buses'] == bus)[0][0]
        load_idx = np.where(env.grid['load_buses'] == bus)[0][0]
        P_load_on_gen[gen_idx] = actual_load_p[load_idx]
        Q_load_on_gen[gen_idx] = actual_load_q[load_idx]

    S_gen = S_HE_mw[env.grid['gen_buses']] + (P_load_on_gen + 1j * Q_load_on_gen)
    gen_p_mw = np.real(S_gen)
    gen_q_mva = np.imag(S_gen)

    state['gen'][:, 0] = np.real(S_gen[env.grid['non_slack_indices']])
    state['gen'][:, 1] = np.imag(S_gen[env.grid['non_slack_indices']])

    slack_p = np.real(S_gen[env.grid['slack_idx']])
    slack_q = np.imag(S_gen[env.grid['slack_idx']])
    state['slack'] = np.array([slack_p, slack_q])

    V_HE_magnitude = np.abs(V_HE)
    V_HE_angle = np.angle(V_HE) 
    state['bus_voltage'][:, 0] = V_HE_magnitude
    state['bus_voltage'][:, 1] = V_HE_angle

    #此时state应该作为下一个状态,但是还没有被y赋值修改

    return V_HE, S_HE, terms, state


if __name__ == "__main__":
    #先把m文件路径找到，并且运行一下case30，读取mpc
    eng = matlab.engine.start_matlab()
    path = eng.eval("which('case30.m')")
    mpc = eng.case30()
    eng.quit()

    env = PowerSystemEnv(mpc, path)
    env.reset()
    net_HE = env.my_net_HE  #给的是变体HE,我先不加帕德近似

    #存储一下bus节点类型,slack为3,PV为2，PQ为1
    bus_types = np.zeros(net_HE.nb, dtype=int)
    bus_types[net_HE.ref] = 3
    bus_types[net_HE.pv] = 2    #不包括松弛节点
    bus_types[net_HE.pq] = 1

    #验证dcv越界来源
    action = np.array(env.action_space.sample())
    next_obs, reward, termination, truncation, info = env.step(action)

    state = env.current_state.copy()

    dcv, info_dcv = env.test_calculate_dcv(env.current_state)
    for key, value in info_dcv.items():
        if key in ['q_non_slack_min', 'q_non_slack_max', 'q_slack_min', 'q_slack_max']:
            print(f"{key}: {value}")
        if key in ['q_non_slack', 'q_slack']:
            print(f'{key}: {value}')
        if key == 'over_limit_indices':
            print(f"{key}: {value}\n")

    #这里应该找出Q越界的节点
    over_limit_indices = info_dcv['over_limit_indices']
    over_limit_indices_include_slack = info_dcv['over_limit_indices_include_slack']     #包含松弛节点
    #所有gen边界
    q_non_slack_min = info_dcv['q_non_slack_min']
    q_non_slack_max = info_dcv['q_non_slack_max']
    q_slack_min = info_dcv['q_slack_min']
    q_slack_max = info_dcv['q_slack_max']
    #整合成整个gen的Q边界,方便索引取值
    q_all_min = np.concatenate([[q_slack_min], q_non_slack_min])
    q_all_max = np.concatenate([[q_slack_max], q_non_slack_max])
    #所有gen真实数值
    q_slack = info_dcv['q_slack']
    q_non_slack = info_dcv['q_non_slack']
    q_all = np.concatenate([[q_slack], q_non_slack])
    #action索引info_dcv
    in_boundary = info_dcv['in_boundary']

    #获取发电机在母线中的索引
    gen_buses = env.grid['gen_buses']
    non_slack_buses = env.grid['gen_buses'][env.grid['non_slack_indices']]

    #找到越界节点后，修改改节点的输入,PV节点修改为PQ节点,这里也只是将越界机组数值提取出来
    over_limit = [] #存储着越界电机的母线索引,以及该母线应该更改成的Q边界
    for i in over_limit_indices:
        bus = non_slack_buses[i]  #这是越界机组母线索引
        bus_types[bus] = 1  #这个节点修改为pq节点
        if q_non_slack[i] < q_non_slack_min[i]:
            over_limit.append((bus, q_non_slack[i], q_non_slack_min[i], 'min'))
        if q_non_slack[i] > q_non_slack_max[i]:
            over_limit.append((bus, q_non_slack[i], q_non_slack_max[i], 'max'))

    print(f'over_limit: {over_limit}\n')    

    #下面进行HE求解,先构建HE入口
    Vsp = np.ones(net_HE.nb)
    S_total = np.zeros(net_HE.nb, dtype=np.complex128)

    #首先获取动作，保持和上一个的动作一致，因为，我们是对上次HE求解不满意，所以重新做一下
    # 解析动作
    p_set = action[:env.grid['num_non_slack']]
    v_set = action[env.grid['num_non_slack']:env.grid['num_non_slack']+env.grid['num_gen']]
    load_shed = action[env.grid['num_non_slack']+env.grid['num_gen']:env.grid['num_non_slack']+env.grid['num_gen']+env.grid['num_loads']]
    wind_curt = action[-env.grid['num_wind']:]

    control = {
        'p_set': p_set,      # 后面要从action里面取出未越界节点,其对应的p需要由action获取
        'v_set': v_set,          # v_set应该包含slack和未越界节点
        'load_shed': load_shed,       # 负荷削减比例
        'wind_curt': wind_curt        # 弃风比例
    }
    #取出负载的真实p,q
    actual_load_p = env.re_HE_load_p
    actual_load_q = env.re_HE_load_q

    #节点分类,前面已经修改过第一次迭代后将pv变成pq的节点类型
    pq_indices = np.where(bus_types == 1)[0]
    pv_indices = np.where(bus_types == 2)[0]
    ref_indices = np.where(bus_types == 3)[0]   #松弛节点不会改变

    #修改Vsp
    ones_array = np.ones(len(in_boundary)).astype(int)
    in_boundary_temp = in_boundary + ones_array
    in_boundary_plus_slack = np.concatenate([[0], in_boundary_temp])   #加上松弛节点
    Vsp[gen_buses[in_boundary_plus_slack]] = (control['v_set'])[in_boundary_plus_slack]

    #修改P_pv
    S_total[net_HE.net.gen.bus[in_boundary]] = (control['p_set'])[in_boundary]

    #修改S_pq
    S_total[net_HE.net.load.bus] = S_total[net_HE.net.load.bus] - actual_load_p - 1j * actual_load_q
    if len(net_HE.net.sgen.bus) > 0:
        S_total[net_HE.net.sgen.bus] = S_total[net_HE.net.sgen.bus] + net_HE.net.sgen.p_mw

    #还要补充PV节点转化过来的PQ节点,之前一直没有处理,新的PQ节点P来自动作,Q来自边界
    #这里已经除去了松弛节点
    q_list = [item[1] for item in over_limit]
    type_list = [item[3] for item in over_limit]
    #下面这两项重要
    boundary_list = [item[2] for item in over_limit]
    bus_list = [item[0] for item in over_limit]
    p_PQ = [x for x in p_set if x not in p_set[in_boundary]]    #获取了p_set中其他越限节点的p_set,传给PQ节点的P

    #转化成array,方便处理数据
    boundary_list = np.array(boundary_list)
    p_PQ = np.array(p_PQ)
    #松弛节点不转化成PQ节点,这就意味着,仍未PV,那么只需要提供V,但是P没有提供,不用提供

    #接下来把新的PQ节点数据加进S_pq
    S_total[bus_list] = S_total[bus_list] + p_PQ + 1j * boundary_list
    S_total = S_total/net_HE.net._ppc["baseMVA"]

    #提取数据
    P_pv = np.real(S_total[pv_indices])
    S_pq = S_total[pq_indices]

    net_HE.pv = pv_indices
    net_HE.pq = pq_indices
    net_HE.npv = net_HE.pv.shape[0]

    # print(f"len(net_HE.pq) = {len(net_HE.pq)}")
    # print(f"len(net_HE.pv) = {len(net_HE.pv)}")
    # print(f"npv:{net_HE.npv}")
    # print(f"len(S_pq) = {len(S_pq)}")
    # print(f"len(P_pv) = {len(P_pv)}")
    # print(f"Vsp={len(Vsp)}")

    net_HE.update_Y_LHS_dynamic()

    V_HE,S_HE,terms = net_HE.run_DHE(S_pq, P_pv, Vsp)

    S_HE_mw = S_HE * net_HE.net._ppc["baseMVA"]

    load_on_gen_buses = np.intersect1d(env.grid['load_buses'], env.grid['gen_buses'])

    P_load_on_gen = np.zeros(env.grid['num_gen'])
    Q_load_on_gen = np.zeros(env.grid['num_gen'])

    for bus in load_on_gen_buses:
        gen_idx = np.where(env.grid['gen_buses'] == bus)[0][0]
        load_idx = np.where(env.grid['load_buses'] == bus)[0][0]
        P_load_on_gen[gen_idx] = actual_load_p[load_idx]
        Q_load_on_gen[gen_idx] = actual_load_q[load_idx]

    S_gen = S_HE_mw[env.grid['gen_buses']] + (P_load_on_gen + 1j * Q_load_on_gen)
    gen_p_mw = np.real(S_gen)
    gen_q_mva = np.imag(S_gen)

    V_HE_magnitude = np.abs(V_HE)
    V_HE_angle = np.angle(V_HE) 
    state['bus_voltage'][:, 0] = V_HE_magnitude
    state['bus_voltage'][:, 1] = V_HE_angle

    state['gen'][:, 0] = np.real(S_gen[env.grid['non_slack_indices']])
    state['gen'][:, 1] = np.imag(S_gen[env.grid['non_slack_indices']])

    slack_p = np.real(S_gen[env.grid['slack_idx']])
    slack_q = np.imag(S_gen[env.grid['slack_idx']])
    state['slack'] = np.array([slack_p, slack_q])

    dcv, info_dcv = env.test_calculate_dcv(state)
    for key, value in info_dcv.items():
        if key in ['q_non_slack', 'q_slack']:
            print(f'PQ转化后{key}: {value}')
        if key == 'over_limit_indices':
            print(f"PQ转化后{key}: {value}\n")

    


