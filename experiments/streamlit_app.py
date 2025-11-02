import os
import sys
# # 设置 SUMO_HOME 为仓库内的 sumo 文件夹
# repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sumo_home = os.path.join(repo_root, "sumo")
# 如果环境变量不存在，可以手动设置
if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = "sumo-home"
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
import subprocess
import streamlit as st
from simulation import simulation_start

# --- SUMO 环境检查 ---
def init_sumo_env():
    if "SUMO_HOME" in os.environ:
        print(os.environ["SUMO_HOME"] )
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        sys.path.append(tools)
    else:
        sys.exit("Please declare the environment variable 'SUMO_HOME'")

# --- 文件上传逻辑 ---
def handle_file_uploads(net_uploader, rou_uploader, additional_uploads):
    tmpdir = tempfile.mkdtemp(prefix="tlab_")
    net_path = None
    rou_paths = {}
    additional_paths = {}

    # NET 文件
    if net_uploader:
        net_path = os.path.join(tmpdir, Path(net_uploader.name).name)
        with open(net_path, 'wb') as f:
            f.write(net_uploader.getbuffer())

    # rou 文件（同名覆盖）
    for fup in rou_uploader:
        filename = Path(fup.name).name
        p = os.path.join(tmpdir, filename)
        with open(p, 'wb') as f:
            f.write(fup.getbuffer())
        rou_paths[filename] = p  # 后上传覆盖前一个

    # ADDITIONAL 文件（同名覆盖）
    for fup in additional_uploads:
        filename = Path(fup.name).name
        p = os.path.join(tmpdir, filename)
        with open(p, 'wb') as f:
            f.write(fup.getbuffer())
        additional_paths[filename] = p  # 后上传覆盖前一个

    return net_path, list(rou_paths.values()), list(additional_paths.values())


# --- 解析 NET 文件中的信号灯及其坐标 ---
def parse_tls_positions(net_path):
    """优先解析 <tlLogic> 获取信号灯 id；尝试用同 id 的 <junction> 获取坐标；若找不到则回退到第一个 traffic_light junction 的坐标或 (0,0)。"""
    tls_nodes = []
    try:
        tree = ET.parse(net_path)
        root = tree.getroot()
        # 收集所有 tlLogic id
        tl_elems = root.findall('.//tlLogic')
        junctions = {j.get('id'): j for j in root.findall('.//junction') if j.get('id')}

        if tl_elems:
            for tl in tl_elems:
                tid = tl.get('id')
                x = y = None
                # 尝试在 junctions 中找到同 id 的 junction
                j = junctions.get(tid)
                if j is not None:
                    try:
                        x = float(j.get('x'))
                        y = float(j.get('y'))
                    except Exception:
                        x = y = None
                # 如果仍然没有坐标，尝试找到任意 traffic_light junction
                if x is None or y is None:
                    any_j = next((jj for jj in root.findall('.//junction') if jj.get('type') == 'traffic_light' and jj.get('x') and jj.get('y')), None)
                    if any_j is not None:
                        try:
                            x = float(any_j.get('x'))
                            y = float(any_j.get('y'))
                        except Exception:
                            x = y = 0.0
                    else:
                        x = y = 0.0
                tls_nodes.append({"id": tid, "x": x, "y": y})
        else:
            # 回退：直接使用 junction 的 id
            for j in root.findall('.//junction'):
                if j.get('type') == 'traffic_light':
                    jid = j.get('id')
                    try:
                        x = float(j.get('x'))
                        y = float(j.get('y'))
                    except Exception:
                        x = y = 0.0
                    tls_nodes.append({"id": jid, "x": x, "y": y})
    except Exception as e:
        st.error(f"解析 net 文件失败：{e}")
    return tls_nodes

# --- 显示信号灯列表并返回用户选择的信号灯 ---
def show_map_and_select_tls(tls_nodes):
    if not tls_nodes:
        st.warning("未在 NET 文件中解析到信号灯")
        return []
    st.write("解析到以下信号灯及坐标：")
    for n in tls_nodes:
        st.write(f"- {n['id']}  (x={n['x']:.2f}, y={n['y']:.2f})")
    # 使用多选框选择信号灯 ID
    selected_tls = st.multiselect("请选择要控制的信号灯 ID（可多选）", [n['id'] for n in tls_nodes])
    return selected_tls

# 新增：在本地启动 netedit 的辅助函数
def launch_netedit(netedit_path: str, net_file: str):
    try:
        if not netedit_path:
            st.error("请提供 netedit 可执行文件路径。")
            return False
        if not os.path.isfile(netedit_path):
            st.error(f"未找到 netedit 可执行文件：{netedit_path}")
            return False
        # 使用 Popen 启动本地 netedit（仅在运行 Streamlit 的本机上有效）
        subprocess.Popen([netedit_path, net_file], cwd=os.path.dirname(netedit_path))
        st.success("已在本地启动 netedit（请在本机查看窗口）。")
        return True
    except Exception as e:
        st.error(f"启动 netedit 失败：{e}")
        return False

# --- Streamlit 页面 ---
def main():
    # 大标题，自定义字号
    st.markdown('<h1 style="text-align: center; font-size:48px; color:#0A74DA;">东南大学交通仿真大模型</h1>', unsafe_allow_html=True)

    st.set_page_config(page_title="东南大学交通仿真大模型", layout="wide")
    # --- 左侧侧边栏 ---
    st.sidebar.title("配置区")
    api_key = st.sidebar.text_input("大模型 API Key", type="password")
    model_name = st.sidebar.text_input("模型名称")

    # 侧边栏：netedit 可执行路径（可为空，若为空则无法启动本地 netedit）
    default_netedit = ""
    if "SUMO_HOME" in os.environ:
        default_netedit = os.path.join(os.environ["SUMO_HOME"], "bin", "netedit.exe")
    netedit_path = st.sidebar.text_input("netedit 可执行文件路径（可选，用于在本地打开路网）", value=default_netedit)

    # --- 中间文件上传 ---
    st.markdown("#### 文件上传与信号灯选择")
    net_uploader = st.file_uploader("上传 NET 文件 (必选)", type=["xml"], key="net")
    rou_uploader = st.file_uploader("上传 ROU or FLOWS 文件（必选, 支持多文件）", type=["xml"], accept_multiple_files=True, key="rou or flows")
    additional_uploads = st.file_uploader("上传 ADDITIONAL 文件（可选, 支持多文件）", type=["xml"], accept_multiple_files=True, key="add")
    submit_btn = st.button("提交文件并显示地图")

    if "submitted" not in st.session_state:
        st.session_state["submitted"] = False

    if submit_btn and net_uploader and rou_uploader:
        st.session_state["submitted"] = True
        st.session_state["net_path"], st.session_state["rou_paths"], st.session_state["additional_paths"] = handle_file_uploads(net_uploader, rou_uploader, additional_uploads)
        st.session_state["tls_nodes"] = parse_tls_positions(st.session_state["net_path"])

    if st.session_state["submitted"]:
        tls_nodes = st.session_state.get("tls_nodes", [])

        # 提供按钮：在本地用 netedit 打开已保存的 net 文件
        st.markdown("#### 本地打开（可选）")
        if st.button("在本地用 netedit 打开路网"):
            net_file_to_open = st.session_state.get("net_path")
            if net_file_to_open:
                launch_netedit(netedit_path, net_file_to_open)
            else:
                st.error("未找到已保存的 net 文件。")

        selected_tls = show_map_and_select_tls(tls_nodes)
        if selected_tls:
            st.success(f"已选择信号灯: {selected_tls}")
            st.subheader("智能体与方案")
            agent_type = st.selectbox("选择智能体", ["ACAgent", "QLAgent"])

            fusion_ratio,llm_score_weight = 0,0
            if agent_type == "ACAgent":
                scheme = st.selectbox(
                    "选择方案",
                    [
                        "固定时序控制",
                        "决策融合控制",
                        "语义先导模式",
                        "相位甄别模式",
                        "评分融合决策"
                    ]
                )
                if scheme == "固定时序控制":
                    llm_value = 0
                elif scheme == "决策融合控制":
                    llm_value = 1
                    fusion_ratio = st.number_input("请输入融合比例（LLM 权重，0–1）", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
                elif scheme == "语义先导模式":
                    llm_value = 2
                elif scheme == "相位甄别模式":
                    llm_value = 3
                elif scheme == "评分融合决策":
                    llm_value = 4
                    llm_score_weight = st.number_input("请输入 LLM 评分的融合权重（0–1）", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
            elif agent_type == "QLAgent":
                scheme = st.selectbox(
                    "选择方案",
                    [
                        "固定时序控制",
                        "决策融合控制",
                        "语义先导模式",
                        "相位甄别模式",
                    ]
                )
                if scheme == "固定时序控制":
                    llm_value = 0
                elif scheme == "决策融合控制":
                    llm_value = 1
                    fusion_ratio = st.number_input("请输入融合比例（LLM 权重，0–1）", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
                elif scheme == "语义先导模式":
                    llm_value = 2
                elif scheme == "相位甄别模式":
                    llm_value = 3

            st.subheader("大模型交互区")
            chat_input = st.text_area(
                "输入对大模型的管控指令（可为空）",
                value=st.session_state.get("chat_input", ""),
                height=150
            )
            st.session_state["chat_input"] = chat_input
            # 开始仿真按钮
            # --- 开始仿真按钮（修改） ---
            if st.button("开始仿真"):
                net_file = st.session_state.get("net_path")
                rou_files = st.session_state.get("rou_paths")
                additional_files = st.session_state.get("additional_paths", [])
                rl_tls_ids = selected_tls
                Agents = agent_type
                chat_input = st.session_state.get("chat_input", "")

                try:
                    st.info("正在启动仿真，请稍候...")
                    # 按照约定位置参数调用 simulation_start，使用关键字参数避免参数重复
                    simulation_start(
                        model_name=model_name,
                        API_KEY=api_key,
                        llm=llm_value,
                        NET_FILE=net_file,
                        ROU_FILE=rou_files,
                        ADDITIONAL_FILES=additional_files,
                        rl_tls_ids=rl_tls_ids,
                        Agents=Agents,
                        fusion_ratio=fusion_ratio,
                        llm_score_weight=llm_score_weight,
                        chat_input=chat_input,
                    )
                    st.success("仿真已启动（simulation_start 已返回）。")

                except Exception as e:
                    # 🚨 关键改动：仿真失败只提示，不清空 session_state
                    st.error(f"启动仿真失败：{e}")
                    st.info("请检查文件或参数后修改再重新运行。")
                    # 可选：把运行结果缓存清空，但保留文件和选项
                    # st.session_state['sim_result'] = None

if __name__ == "__main__":
    init_sumo_env()
    main()
