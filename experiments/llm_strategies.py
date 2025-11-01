import numpy as np

def run_llm0(env, rl_agents):
    done = {"__all__": False}
    while not done["__all__"]:
        print("仅使用强化学习进行决策")
        print(f"当前仿真时间（秒）：{env.sim_step}")
        actions = {ts: rl_agents[ts].act() for ts in rl_agents.keys()}
        print(f"Actions : {actions}")
        #s为observation，r为reward, done为是否结束，info为一些统计信息
        s, r, done, info = env.step(action=actions)
        # print(f"state : {s['219']}")
        print(f"reward : {r}")
        # print(f"Rewards : {r}")
        # print(f"Next Observation : {s}")
        for ts in rl_agents.keys():
            rl_agents[ts].learn(next_state=env.encode(s[ts], ts), reward=r[ts],final_action=actions[ts])


def run_llm1(env, rl_agents, advisor, args):
    done = {"__all__": False}
    while not done["__all__"]:
        print("使用强化学习与LLM融合决策，方案一：按比例融合")
        print(f"\n当前仿真时间: {env.sim_step}")
        rl_actions = {ts: rl_agents[ts].act() for ts in rl_agents.keys()}
        llm_actions = advisor.get_suggestions(env=env,args=args,tls_ids=rl_agents.keys())
        if llm_actions:
            actions = {
                ts: rl_actions[ts] if np.random.rand() < (1 - args.fusion_ratio) else llm_actions[ts]
                for ts in rl_agents.keys()
            }
        else:
            actions = rl_actions
        print(f"rl_actions: {llm_actions}, llm_action: {llm_actions},actions: {actions}")
        s, r, done, _ = env.step(action=actions)
        for ts in rl_agents.keys():
            rl_agents[ts].learn(env.encode(s[ts], ts), r[ts], actions[ts])

def run_llm2(env, rl_agents, advisor, args):
    while not done["__all__"]:
        print("使用强化学习与LLM融合决策，方案二：LLM建议集合")
        print(f"\n当前仿真时间: {env.sim_step}")
        accepted_actions = advisor.get_suggestions(env=env,args=args,tls_ids=rl_agents.keys())
        if accepted_actions:
            rl_actions = {ts: rl_agents[ts].act(accepted_action=accepted_actions[ts]) for ts in rl_agents.keys()}
        else:
            rl_actions = {ts: rl_agents[ts].act() for ts in rl_agents.keys()}
        actions = rl_actions
        print(f"accepted: {accepted_actions}, rl: {rl_actions}")
        s, r, done, _ = env.step(action=rl_actions)
        for ts in rl_agents.keys():
            rl_agents[ts].learn(env.encode(s[ts], ts), r[ts], rl_actions[ts])

def run_llm3(env, rl_agents, advisor, args):
    while not done["__all__"]:
        print("使用强化学习与LLM融合决策，方案三：拒绝采纳")
        print(f"\n当前仿真时间: {env.sim_step}")
        tls_ids=list(rl_agents.keys())
        tls_info = env.get_tls_info(tls_ids)
        accepted_actions = {ts: list(range(tls_info[str(ts)]["相位数量"])) for ts in rl_agents.keys()}
        actions = {}
        actions_first = {}
        i = 0
        while True:
            i += 1
            if tls_ids == []:
                break

            rl_actions = {ts: rl_agents[ts].act(accepted_action=accepted_actions[ts]) for ts in tls_ids}
            llm_judge = advisor.get_suggestions(env=env,args=args,tls_ids=tls_ids,rl_actions=rl_actions)
            if llm_judge is None:
                print("LLM未返回建议，强制采纳强化学习的建议")
                for ts in tls_ids:
                    actions[ts] = rl_actions[ts]
                break
            if i==1:
                actions_first = rl_actions.copy()
            print(f"rl: {rl_actions}, judge: {llm_judge}")
            for ts in tls_ids:
                if llm_judge[ts] == 1:
                    actions[ts] = rl_actions[ts]
                    tls_ids.remove(ts)
                else:
                    accepted_actions[ts].remove(rl_actions[ts])
                    if accepted_actions[ts] == []:
                        print(f"信号灯 {ts} 的可选相位已耗尽，强制采纳强化学习的建议相位 {actions_first[ts]}")
                        actions[ts] = actions_first[ts]
                        tls_ids.remove(ts)
        print(f"final: {actions}")
        s, r, done, _ = env.step(action=actions)
        for ts in rl_agents.keys():
            rl_agents[ts].learn(env.encode(s[ts], ts), r[ts], actions[ts])

def run_llm4(env, rl_agents, advisor, args):
    done = {"__all__": False}
    while not done["__all__"]:
        print("使用强化学习与LLM融合决策，方案四：llm评分")
        print(f"当前仿真时间（秒）：{env.sim_step}")
        ac_actions = {ts: rl_agents[ts].act() for ts in rl_agents.keys()}
        actions = ac_actions
        s, r, done, info = env.step(action=actions)
        print(f"Actions : {actions}")
        # 每个 agent 执行一次学习，传入 llm_score
        llm_scores = advisor.get_suggestions(env=env,args=args,tls_ids=rl_agents.keys(),rl_actions=ac_actions)
        if llm_scores is None:
            llm_scores = {ts: 0.5 for ts in rl_agents.keys()}  # 如果 LLM 没有返回评分，默认评分为 0.5
        print(f"llm_scores : {llm_scores}")
        for ts in rl_agents.keys():
            rl_agents[ts].learn(next_state=env.encode(s[ts], ts), reward=r[ts], final_action=actions[ts],llm_score=llm_scores[ts])
