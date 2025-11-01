from config import get_args
# from deepSeekAdvisor import DeepSeekAdvisor
from my_env import SumoEnvironment  # 假设你有一个自定义的 SumoEnvironment 类
from rl_agents import setup_agents
from llm_strategies import run_llm0,run_llm1, run_llm2, run_llm3, run_llm4
from deepSeekAdvisor import DeepSeekAdvisor
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

def simulation_start(model_name,API_KEY,llm,NET_FILE,ROU_FILE,ADDITIONAL_FILES,rl_tls_ids,Agents,fusion_ratio=0,llm_score_weight=0,add_instruction=None,chat_input=None):
    args = get_args()
    args.llm = llm
    args.model_name = model_name
    args.API_KEY = API_KEY
    args.fusion_ratio = fusion_ratio
    args.llm_weight = llm_score_weight
    args.user_input = chat_input
    args.seconds = 3600
    args.min_green = 10
    args.max_green = 60

    args.gui = True
    args.fixed = False
    if args.fixed:
        rl_tls_ids=[]
        out_csv = (
        rf"固定信号配时\\"
        rf"固定信号配时"
            )
    else:
        if args.llm != 0:
            advisor = DeepSeekAdvisor(args)
        out_csv = (
        rf"LLM{args.llm}\\"
        rf"alpha{args.alpha}_gamma{args.gamma}_eps{args.epsilon}_decay{args.decay}_reward{args.reward}_s{args.seconds}_p{args.fusion_ratio}"
            )
    #传入需要强化学习控制的信号灯编号，若为空则默认所有信号灯均为固定配时
    #reward_fn定义奖励函数，["diff-waiting-time","average-speed","queue","pressure"]
    #配套权重reward_weights，[]
    reward_fn = ["diff-waiting-time","average-speed","queue"]
    reward_weights = [0.33,0.33,0.33]
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
        add_instruction=add_instruction,
    )
    if args.rl:
        rl_tls_ids=env.rl_tls_ids
    rl_agents = setup_agents(env=env, args=args,initial_states=env.reset(),choose_agents=Agents,tls_ids=rl_tls_ids)
    done = {"__all__": False}
    if args.fixed:
        done = {"__all__": False}
        while not done["__all__"]:
            print("Step:", env.sim_step)
            _, _, done, _ = env.step({})
    else:
        if args.llm ==0:
            run_llm0(env, rl_agents)
        elif args.llm == 1:
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



if __name__ == "__main__":
    model_name = "Qwen/Qwen3-32B"
    API_KEY = "sk-rxhwvnjkaofyitwacnspbrobslqpuvcyjbxfdwnmmwvulliv"



    # llm取值：0-仅强化学习控制，1-方案一，2-方案二，3-方案三，4-方案四
    # 方案一：强化学习与大模型概率融合，需要定义fusion_ratio的取值
    # 方案二：大模型预先给出相位的候选集合，由强化学习进行学习
    # 方案三：大模型拒绝或接受强化学习给出的相位，循环往复
    # 方案四：仅限大模型进行AC智能体，大模型与AC智能体一起参与动作评分过程
    llm = 0
    NET_FILE = rf"net\bologna\acosta\acosta_buslanes.net.xml"
    ROU_FILE = [rf"net\bologna\acosta\acosta.rou.xml",
                rf"net\bologna\acosta\acosta_busses.rou.xml",
                rf"net\bologna\acosta\new_add.rou.xml"
                ]



    ADDITIONAL_FILES = [rf"net\bologna\acosta\acosta_bus_stops.add.xml",
    rf"net\bologna\acosta\acosta_detectors.add.xml",
    rf"net\bologna\acosta\acosta_tls.add.xml",
    rf"net\bologna\acosta\acosta_vtypes.add.xml",
    ]

    rl_tls_ids = ["219","220","273"]
    # 可选ACAgent或QLAgent
    Agents = "QLAgent"
    fusion_ratio = 1
    
    chat_input = "对于救护车等紧急车辆请优先保证其通行"


    simulation_start(model_name,API_KEY,llm,NET_FILE,ROU_FILE,ADDITIONAL_FILES,rl_tls_ids,Agents,fusion_ratio=fusion_ratio,chat_input=chat_input)
