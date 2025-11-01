# config.py
import argparse

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Q-Learning Single-Intersection"
    )
    # 基础超参数
    parser.add_argument("-a", dest="alpha", type=float, default=0.1, help="Alpha learning rate.")
    parser.add_argument("-g", dest="gamma", type=float, default=0.99, help="Gamma discount rate.")
    parser.add_argument("-e", dest="epsilon", type=float, default=0.05, help="Epsilon.")
    parser.add_argument("-me", dest="min_epsilon", type=float, default=0.005, help="Minimum epsilon.")
    parser.add_argument("-d", dest="decay", type=float, default=1.0, help="Epsilon decay.")
    parser.add_argument("-mingreen", dest="min_green", type=int, default=10, help="Minimum green time.")
    parser.add_argument("-maxgreen", dest="max_green", type=int, default=30, help="Maximum green time.")
    parser.add_argument("-s", dest="seconds", type=int, default=2000, help="Number of simulation seconds.")
    parser.add_argument("-r", dest="reward", type=str, default="wait", help="Reward function: [queue|wait].")
    parser.add_argument("-runs", dest="runs", type=int, default=1, help="Number of runs.")

    # 模式选项
    parser.add_argument("-gui", action="store_true", default=True, help="Run SUMO with GUI.")
    parser.add_argument("-fixed", action="store_true", default=False, help="Run with fixed traffic signals.")
    parser.add_argument("-rl",action="store_true", default=False,help="Run with all traffic signals controlled by reinforcement learning.")
    parser.add_argument("-API_KEY", type=str, default="sk-rxhwvnjkaofyitwacnspbrobslqpuvcyjbxfdwnmmwvulliv", help="API key.")
    parser.add_argument("-llm", type=int, choices=[0, 1, 2, 3], default=0, help="LLM mode (0: RL only, 1-3: LLM variants).")
    parser.add_argument("-p", dest="fusion_ratio", type=float, default=0, help="Fusion ratio RL vs LLM actions.")
    parser.add_argument("-user_input", dest="user_input", type=str, default=None, help="user input for LLM.")
    parser.add_argument("-model_name", type=str, default="Qwen/Qwen3-32B", help="Model name.")
    parser.add_argument("-w", dest="llm_weight", type=float, default=0, required=False, help="weight between LLM and AC.\n")

    return parser.parse_args()
