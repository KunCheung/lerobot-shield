# Grab Things

这个目录是 Shield 示例里的 **两轮机器人抓取 notebook** 工程。它基于 LeRobot 的异步 policy 推理、RoboCrew 的 LLM 工具调用框架、SOFollower 右臂控制链，以及 OpenCV 摄像头输入。
具体模型策略为"Grigorij/act_right-arm-grab-notebook-2"

当前目标很明确：机器人靠近桌面上的 notebook，切到近距离视角，用 **右臂** 执行 notebook 抓取 policy，抓住后停止任务并保持夹爪力矩。左臂不参与抓取，头部只用于视角辅助。

## 适用硬件

默认配置面向这台机器人：

- 底盘：两轮差速底盘，只支持 `move_forward`、`move_backward`、`turn_left`、`turn_right`
- 机械臂：左右双臂，但 notebook 抓取只使用右臂
- 头部：用于普通视角和近距离 precision 视角切换
- 主相机：`MAIN_CAMERA_ID = 1`
- 右臂相机：`RIGHT_ARM_CAMERA_ID = 2`
- 右臂 + 底盘串口：`COM4`
- 左臂 + 头部串口：`COM5`
- 两轮底盘电机 ID：左轮 `9`，右轮 `10`

这个目录不是三轮 XLeRobot 的通用版本，不包含横移逻辑，也不要用写死 `/dev/arm_right` 的默认 RoboCrew 录姿态脚本。

## 文件说明

- `grab_notebook.py`
  主流程脚本。LLM 负责寻找和靠近 notebook；到达抓取距离后调用 `Grab_a_notebook` 工具，由 VLA policy 控制右臂完成抓取。

- `record_right_arm_cobra.py`
  右臂 `cobra` 起始姿态录制脚本。它只连接 `COM4` 上的右臂 1-6 号舵机，并保存 `~/.cache/robocrew/positions/cobra.json`。

- `debug_right_arm_extension.py`
  右臂执行链调试脚本。它不依赖 LLM、相机或 policy server，只复现 `grab_notebook.py` 抓取阶段使用的 `SOFollower(COM4)` 控制链，用于检查右臂是否能跟随 policy-like 目标姿态，以及是否出现 overload。

## 必要配置

### Python 环境

使用本项目的 `lerobot` conda 环境：

```powershell
conda activate lerobot
```

如果直接指定解释器，可使用：

```powershell
C:\softwares\miniconda\envs\lerobot\python.exe
```

### LLM API

`grab_notebook.py` 默认使用：

```python
MODEL_NAME = "openai:kimi-k2.6"
MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1"
```

脚本会读取上一层的环境文件：

```text
examples/shield/.env
```

至少需要配置：

```env
MOONSHOT_API_KEY=你的 Moonshot API Key
```

脚本会自动设置：

```text
OPENAI_BASE_URL=https://api.moonshot.cn/v1
```

### Policy Server

VLA 抓取通过 LeRobot async inference policy server 执行。默认地址：

```python
POLICY_SERVER_ADDRESS = "127.0.0.1:8080"
POLICY_NAME = "Grigorij/act_right-arm-grab-notebook-2"
POLICY_TYPE = "act"
POLICY_DEVICE = "cpu"
```

如果 policy 需要 Hugging Face 权限，请先确保本机已经登录或配置好 token。

### 校准文件

脚本会从两轮机器人校准文件提取右臂校准：

```text
~/.cache/huggingface/lerobot/calibration/robots/xlerobot_2wheels/my_xlerobot_2wheels_lab.json
```

然后同步为 `SOFollower` 右臂校准：

```text
~/.cache/huggingface/lerobot/calibration/robots/so_follower/right_arm.json
```

如果缺少 `xlerobot_2wheels` 校准文件，`grab_notebook.py`、`record_right_arm_cobra.py` 和 `debug_right_arm_extension.py` 都会失败。

### 右臂 cobra 姿态

`grab_notebook.py` 在启动 VLA 前必须找到：

```text
~/.cache/robocrew/positions/cobra.json
```

这个文件必须是右臂专用姿态：

```json
{
  "arm_side": "right",
  "positions": {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
    "gripper": 0.0
  }
}
```

实际数值由 `record_right_arm_cobra.py` 录制，不要手写。缺少这个文件时，抓取工具会明确报错并拒绝从随机姿态启动 policy。

## 运行流程

### 1. 录制右臂 cobra 起始姿态

先确认 `COM4` 没有被其他程序占用，然后运行：

```powershell
C:\softwares\miniconda\envs\lerobot\python.exe .\examples\shield\grab_things\record_right_arm_cobra.py
```

默认流程：

1. 连接 `COM4` 上的右臂 1-6 号舵机
2. 释放右臂 torque
3. 手动把右臂摆到 notebook policy 的 `cobra` 起始姿态
4. 按 Enter 保存

成功后会看到类似输出：

```text
Saved right-arm pose: C:\Users\<you>\.cache\robocrew\positions\cobra.json
```

如果右臂已经在目标姿态，并且不想释放 torque，可以运行：

```powershell
C:\softwares\miniconda\envs\lerobot\python.exe .\examples\shield\grab_things\record_right_arm_cobra.py --keep-torque
```

如需保存后回放验证，可以加：

```powershell
C:\softwares\miniconda\envs\lerobot\python.exe .\examples\shield\grab_things\record_right_arm_cobra.py --verify-recall
```

`--verify-recall` 会重新发送该姿态，可能造成右臂运动，实机旁边要留出安全空间。

### 2. 启动 policy server

新开一个 PowerShell 窗口：

```powershell
conda activate lerobot
python -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080
```

保持这个窗口运行。`grab_notebook.py` 会作为 client 连接它，并把 policy 名称、policy 类型、设备和机器人特征发送过去。

### 3. 运行 notebook 抓取任务

再开一个 PowerShell 窗口：

```powershell
conda activate lerobot
python .\examples\shield\grab_things\grab_notebook.py
```

主流程会：

1. 打开主相机
2. 连接 `COM4` 右臂 + 两轮底盘、`COM5` 左臂 + 头部
3. 让 LLM 寻找并靠近 notebook
4. 到达近距离后切换 precision 视角
5. 调用 `Grab_a_notebook`
6. 右臂移动到 `cobra` 起始位
7. VLA policy 控制右臂抓 notebook
8. 成功后停止任务，不再继续移动
9. 默认保持右臂最终抓取姿态和夹爪力矩，不回 default

## 调试右臂执行链

如果 notebook 抓取失败，先单独确认右臂执行链：

```powershell
conda activate lerobot
python .\examples\shield\grab_things\debug_right_arm_extension.py
```

这个脚本不会启动 LLM、相机或 policy server。它会直接向右臂发送几组单关节和 policy-like 目标姿态，并记录：

- 实际位置是否跟随目标
- `shoulder_lift`、`elbow_flex`、`wrist_flex` 是否出现 overload
- 原始寄存器里的 `Goal_Position`、`Present_Position`、`Present_Load`、`Present_Current`、`Status`、`Torque_Enable`

输出目录：

```text
examples/shield/tmp_images/right_arm_extension_debug/<run_id>/
```

关键文件：

- `session.json`
- `command_events.jsonl`
- `raw_register_events.jsonl`
- `summary.json`

## 抓取调试输出

`grab_notebook.py` 会持续写入：

```text
examples/shield/tmp_images/
```

常用文件：

- `latest_raw_camera.png`
- `latest_raw_camera.json`
- `latest_llm_input.jpg`
- `latest_llm_input.json`

每次 VLA 抓取会生成：

```text
examples/shield/tmp_images/vla_grab_debug/<run_id>/
```

关键文件：

- `session.json`
- `events.jsonl`
- `action_events.jsonl`
- `action_summary.json`

如果 `action_summary.json` 里有 `action_count`，说明 policy server 已经产生动作；如果右臂不动，优先检查 `COM4` 是否被占用、右臂校准是否正确、`SOFollower` 是否能正常连接。

## 重要默认行为

- `KEEP_GRASP_AFTER_VLA = True`
  VLA 结束后右臂断开时不关闭 torque，夹爪继续保持。

- `RESET_ARMS_AFTER_GRAB = False`
  抓取结束后不再把双臂打回 default，避免 notebook 掉落。

- `VLA_EXECUTION_TIME_S = 60`
  右臂 VLA policy 默认运行 60 秒。

- `VLA_START_POSE_NAME = "cobra"`
  抓取前必须先移动到右臂 `cobra` 姿态。

## 常见问题

### 报错：Could not connect on port `/dev/arm_right`

这是用了默认 RoboCrew 录姿态脚本导致的。当前 Windows 两轮机器人右臂端口是 `COM4`，不要运行：

```powershell
python -m robocrew.scripts.robocrew_record_positions --arms right --position-name cobra
```

请改用：

```powershell
C:\softwares\miniconda\envs\lerobot\python.exe .\examples\shield\grab_things\record_right_arm_cobra.py
```

### 报错：缺少 `cobra` 姿态

先运行 `record_right_arm_cobra.py`。`grab_notebook.py` 不会从随机右臂姿态启动 notebook policy。

### LLM 继续移动或尝试横移

当前 system prompt 已按两轮差速机器人约束，只允许前进、后退、左转、右转。`Grab_a_notebook` 成功后任务会停止。如果日志里仍出现异常规划，优先检查是否运行的是当前这个 `grab_notebook.py`。

### VLA 有 action 但右臂抓不住

按顺序检查：

1. `debug_right_arm_extension.py` 是否有 overload
2. `cobra.json` 是否是右臂专用姿态，且姿态接近 policy 起点
3. `vla_grab_debug/<run_id>/action_summary.json` 是否有 action 输出
4. 右臂相机 `RIGHT_ARM_CAMERA_ID = 2` 是否真的看到 notebook 和夹爪
5. 抓取后是否保持 torque，确认 `KEEP_GRASP_AFTER_VLA = True`
