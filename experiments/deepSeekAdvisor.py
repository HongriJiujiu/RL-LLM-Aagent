import traci
import json
from typing import Dict, List
import numpy as np
import json
import requests

class DeepSeekAdvisor:

    def __init__(self, args):
        self.model_name = args.model_name
        self.API_KEY = args.API_KEY

    def default_dump(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    def build_output_format(self,args):
        if args.llm == 1:
            output_format = {
            "role": "user",
            "content": ("请严格按照以下格式输出：\n"
                        "- 输出必须为 **纯 JSON 格式**，字典结构。\n"
                        "- 每个键是信号灯编号（字符串），值为信号灯相位。\n"
                        "- 严格按照此格式回复，**不要添加注释、说明、换行或额外文本**。\n"
                        "- 示例输出：{\"t\": 0}\n\n")}
        elif args.llm == 2:
            output_format = {
            "role": "user",
            "content": ("请严格按照以下格式输出：\n"
                        "- 输出必须为 **纯 JSON 格式**，字典结构。\n"
                        "- 每个键是信号灯编号（字符串），值是一个包含整数的列表，表示可接受的相位集合（相位为整数）；请不要对相位做太过严格的筛选。\n"
                        "- 严格按照此格式回复，**不要添加注释、说明、换行或额外文本**。\n"
                        "- 示例输出：{\"t\": [0,1,2]}\n\n"
                        "如有不确定，默认所有的相位都可以接受。\n")}
        elif args.llm == 3:
            output_format = {
            "role": "user",
            "content": ("请严格按照以下格式输出：\n"
                        "- 输出必须为 **纯 JSON 格式**，字典结构。\n"
                        "- 每个键是信号灯编号（字符串），值为整数0或1，0表示不同意强化学习模型给出的策略，1表示同意强化学习模型给出的策略。\n"
                        "- 严格按照此格式回复，**不要添加注释、说明、换行或额外文本**。\n"
                        "- 示例输出：{\"t\": 0}\n\n"
                        "如有不确定，默认为同意。\n")}
        elif args.llm == 4:
            output_format = {
            "role": "user",
            "content": ("请严格按照以下格式输出：\n"
                        "- 每个键是信号灯编号（字符串），值为分数（浮点数，范围在 0~1 之间），评分越高表明越认可。\n"
                        "- 严格按照此格式回复，**不要添加任何注释、说明文字或换行符**。\n"
                        "- 如果对某个信号灯评分不确定，默认值为 0.5。\n"
                        "- 示例输出：{\"0\": 0.2, \"1\": 0.3, ...}\n")}
        return output_format

    def build_task_description(self,args):
        if args.llm == 1:
            task_description = {
            "role": "system",
            "content": ("请你根据当前时刻各个交叉口的详细信息，判断每一个交叉口信号灯的下一个相位。请注意选取的相位范围为大于等于0，小于相位数量。\n")}
        elif args.llm == 2:
            task_description = {
            "role": "system",
            "content": ("请你根据当前时刻各个交叉口的详细信息，为每个信号灯选择至少一个相位构成下一个时间步长内的 **可接受相位列表**（即候选动作）。请注意选取的相位范围为大于等于0，小于相位数量。\n")}
        elif args.llm == 3:
            task_description = {
            "role": "system",
            "content": ("请你根据当前时刻各个交叉口的详细信息，对每一个信号交叉口判断强化学习模型给出的下一个相位是否合理。\n")}
        elif args.llm == 4:
            task_description = {
            "role": "system",
            "content": ("请你根据当前时刻各个交叉口的详细信息，为每个信号灯当前的控制策略进行评分。\n")}
        return task_description

    def build_messages(self,base_info, system_info,output_format, task_description, user_input=None) -> List[Dict]:
        """
        构建 messages 结构
        :param base_info: 公共部分信息
        :param output_format: 对输出格式的具体要求
        :param task_description: 对当前任务的自然语言描述
        :param user_input: 用户额外输入的内容
        """
        messages = []
        # 系统级描述/角色定位（system）
        messages.append(system_info)
        # 当前任务的描述（user）
        messages.append(task_description)
        # 交叉口信息/公共信息 (user)
        messages.extend(base_info)
        # 对输出格式的要求（user）
        messages.append(output_format)
        # 用户额外输入接口
        if user_input:
            messages.append({
                "role": "user",
                "content": user_input
            })
        return messages

    def send_messages(self,messages, temperature=0.7, max_retries=3):
        """
        发送消息到大模型 API，带有最多 max_retries 次重试。
        """
        url = "https://api.siliconflow.cn/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": False
        }

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(url, headers=headers, json=payload)
                response.raise_for_status()  # 检查 HTTP 错误
                res_json = response.json()

                # 打印完整返回
                print(f"✅ 第 {attempt} 次请求成功，LLM 原始响应:\n", res_json)
                generated_text = res_json['choices'][0]['message']['content'].strip()
                if generated_text.startswith("```"):
                    generated_text = generated_text.strip("`")  # 移除 ```
                    lines = generated_text.split("\n", 1)
                    if len(lines) > 1:
                        generated_text = lines[1].strip()
                try:
                    result_dict = json.loads(generated_text)
                    return result_dict
                except json.JSONDecodeError as e:
                    print("❌ JSON解析失败:", e)
                    return None
            except Exception as e:
                print(f"❌ 第 {attempt} 次请求失败: {e}")

        # 所有尝试失败
        print("🚨 所有尝试均失败，返回空响应")
        return None


    def process_response(self, result_dict,tls_ids,args,tls_info) -> Dict[str, dict]:
        if result_dict is None:
            return None
        if args.llm == 1:
            for tls_id in tls_ids:
                if str(tls_id) not in result_dict:
                    print(f"❌ 输出缺少信号灯 {tls_id} 的建议相位，使用默认相位 0")
                    result_dict[str(tls_id)] = 0
                else:
                    if not (0 <= result_dict[str(tls_id)] < tls_info[str(tls_id)]["相位数量"]):
                        print(f"❌ 信号灯 {tls_id} 的建议相位 {result_dict[str(tls_id)]} 超出范围，使用默认相位 0")
                        result_dict[str(tls_id)] = 0
        elif args.llm == 2:
            for tls_id in tls_ids:
                if str(tls_id) not in result_dict:
                    print(f"❌ 输出缺少信号灯 {tls_id} 的建议相位集合，使用默认全相位集合")
                    result_dict[str(tls_id)] = list(range(tls_info[str(tls_id)]["相位数量"]))
                else:
                    if isinstance(result_dict[str(tls_id)], list) and result_dict[str(tls_id)]:
                        if any(phase < 0 or phase >= tls_info[str(tls_id)]["相位数量"] for phase in result_dict[str(tls_id)]):
                            print(f"❌ 信号灯 {tls_id} 的建议相位 {result_dict[str(tls_id)]} 超出范围，使用默认全相位集合")
                            result_dict[str(tls_id)] = list(range(tls_info[str(tls_id)]["相位数量"]))
                    else:
                        print(f"❌ 信号灯 {tls_id} 的建议 {result_dict[str(tls_id)]} 不是非空列表，使用默认全相位集合")
                        result_dict[str(tls_id)] = list(range(tls_info[str(tls_id)]["相位数量"]))
        elif args.llm == 3:
            for tls_id in tls_ids:
                if str(tls_id) not in result_dict:
                    print(f"❌ 输出缺少信号灯 {tls_id} 的建议，默认同意强化学习的策略")
                    result_dict[str(tls_id)] = 1
                else:
                    if result_dict[str(tls_id)] not in [0, 1]:
                        print(f"❌ 大模型给出的建议 {result_dict[str(tls_id)]} 超出范围[0,1]，默认同意强化学习的策略")
                        result_dict[str(tls_id)] = 1
        elif args.llm == 4:
            for tls_id in tls_ids:
                if str(tls_id) not in result_dict:
                    print(f"❌ 输出缺少信号灯 {tls_id} 的评分，默认评分为0.5")
                    result_dict[str(tls_id)] = 0.5
                else:
                    if not (0 <= result_dict[str(tls_id)] <= 1):
                        print(f"❌ 大模型给出的评分 {result_dict[str(tls_id)]} 超出范围[0,1]，默认评分为0.5")
                        result_dict[str(tls_id)] = 0.5
        return result_dict

    def get_suggestions(self,env,args,tls_ids,rl_actions=None) -> Dict[str, int]:
        
        tls_info = env.get_tls_info(tls_ids)
        print(tls_info)
        if rl_actions:
            for ts in rl_actions.keys():
                tls_info[ts]['强化学习给出的下一个相位：'] = rl_actions[ts]
        base_info = [{"role": "user", "content": f"各个信号交叉口的详细信息，请不要胡乱猜测各个信号等间的关系，：{tls_info}"},]
        system_info = {"role": "system", "content": "你是一个交通信号灯控制专家,允许相位不按照相位顺序进行运行。"}
        user_input = args.user_input if args.user_input else None
        output_format = self.build_output_format(args=args)
        task_description = self.build_task_description(args)
        messages = self.build_messages(base_info=base_info,system_info=system_info,output_format=output_format, 
                                       task_description=task_description, user_input=user_input)
        result_dict = self.send_messages(messages)
        return self.process_response(result_dict,tls_ids,args,tls_info)

    def close(self):
        traci.close()


