import pandapower as pp

# 2. 加载IEEE 30节点系统
# 注意：这里是 case30()
net = pp.networks.case30()

# 3. 运行潮流计算
pp.runpp(net)

# 4. 查看结果
print("潮流计算成功收敛:", net.converged)
print("各节点电压幅值 (p.u.):\n", net.res_bus.vm_pu)
print("发电机组有功出力 (MW):\n", net.res_gen.p_mw)