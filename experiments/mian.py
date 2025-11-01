from config import get_args
# from deepSeekAdvisor import DeepSeekAdvisor
from my_env import SumoEnvironment  # 假设你有一个自定义的 SumoEnvironment 类
from rl_agents import setup_agents
from llm_strategies import run_llm0,run_llm1, run_llm2, run_llm3, run_llm4
from deepSeekAdvisor import DeepSeekAdvisor
# python 苏州路网实验\mian.py -fixed -gui -llm 0

import argparse
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from threading import Lock
# python mian.py -fixed
# Ensure SUMO_HOME is set
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

if __name__ == "__main__":
    args = get_args()
    if args.fixed:
        out_csv = (
        rf"E:\Users\35601\Desktop\sumo-rl\acosta\固定信号配时\\"
        rf"alpha{args.alpha}_gamma{args.gamma}_eps{args.epsilon}_decay{args.decay}_reward{args.reward}_s{args.seconds}_p{args.fusion_ratio}"
            )
    else:
        if args.llm != 0:
            advisor = DeepSeekAdvisor(args)
        out_csv = (
        rf"E:\Users\35601\Desktop\sumo-rl\acosta\LLM{args.llm}\\"
        rf"alpha{args.alpha}_gamma{args.gamma}_eps{args.epsilon}_decay{args.decay}_reward{args.reward}_s{args.seconds}_p{args.fusion_ratio}"
            )
#     NET_FILE = "suzhouluwang/network.net.xml"
#     ROU_FILE = "suzhouluwang/routes.rou.xml"
#     ADDITIONAL_FILES = [
#     "suzhouluwang/osm_polygons.add.xml",
#     "suzhouluwang/osm_stops.add.xml",
#     "suzhouluwang/osm_complete_parking_areas.add.xml",
#     "suzhouluwang/osm_parking_rerouters.add.xml",
#     "suzhouluwang/basic.vType.xml",
# ]
    NET_FILE = "nets/acosta/acosta_buslanes.net.xml"
    ROU_FILE = "nets/acosta/acosta.rou.xml"
    ADDITIONAL_FILES = [
    "nets/acosta/acosta_vtypes.add.xml",
    "nets/acosta/acosta_detectors.add.xml",
    "nets/acosta/acosta_bus_stops.add.xml",
    "nets/acosta/acosta_tls.add.xml"
]
    #传入需要强化学习控制的信号灯编号，若为空则默认所有信号灯均为固定配时
    rl_tls_ids = ["219"]
    Agents = ['ACAgent', 'QLAgent']
    choose_agents = Agents[0]  # 选择 'ACAgent' 或 'QLAgent'
    #reward_fn定义奖励函数，["diff-waiting-time","average-speed","queue","pressure"]
    #配套权重reward_weights，[]
    reward_fn = ["diff-waiting-time","average-speed","queue"]
    reward_weights = [0.25,0.25,0.25]
    env = SumoEnvironment(
        rl=args.rl,
        rl_tls_ids=rl_tls_ids,
        net_file=NET_FILE,
        route_file=ROU_FILE,
        additional_files=ADDITIONAL_FILES, 
        out_csv_name=out_csv,
        use_gui=args.gui,
        fixed_ts=args.fixed,
        num_seconds=args.seconds,
        min_green=args.min_green,
        max_green=args.max_green,
        sumo_warnings=False,
        reward_fn=reward_fn,
        reward_weights=reward_weights,
    )
    if args.rl:
        rl_tls_ids=env.rl_tls_ids
    rl_agents = setup_agents(env=env, args=args,initial_states=env.reset(),choose_agents=choose_agents,tls_ids=rl_tls_ids)
    done = {"__all__": False}
    if args.fixed:
        done = {"__all__": False}
        while not done["__all__"]:
            print("Step:", env.sim_step)
            _, _, done, _ = env.step({})
    else:
        if args.llm ==0:
            run_llm0(env, rl_agents)
        if args.llm == 1:
            run_llm1(env, rl_agents, advisor, args)
        elif args.llm == 2:
            run_llm2(env, rl_agents, advisor, args)
        elif args.llm == 3:
            run_llm3(env, rl_agents, advisor, args)
        elif args.llm == 4:
            run_llm4(env, rl_agents, advisor, args)
        else:
            raise ValueError(f"Unsupported llm option: {args.llm}")
    env.save_csv(out_csv,1)
    env.close()