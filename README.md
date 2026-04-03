# LeRobot Shield

`lerobot-shield` 是一个面向 XLerobot 实机使用的 LeRobot fork。

这个仓库把两部分合并到了同一个代码库里：

- **上游 LeRobot**
  - 统一机器人接口
  - 数据集工具
  - 训练 / 评估基础设施
- **XLerobot 集成**
  - `XLerobot`
  - `XLerobot2Wheels`
  - 键盘 / JoyCon / VR 遥操作示例
  - 更安全的 `shield` 退出示例

写得更直白一点：
这是一个“**LeRobot 框架 + XLerobot 硬件接入**”的合并版仓库，目标是既能直接控制 XLerobot，也能继续复用 LeRobot 的工程结构和工具链。

官方 XLeRobot 软件文档可参考：
<https://xlerobot.readthedocs.io/zh-cn/latest/software/index.html>

## 仓库里有什么

与 XLerobot 直接相关的主要入口如下：

- `src/lerobot/robots/xlerobot`
  - 原始 XLerobot 机器人实现
- `src/lerobot/robots/xlerobot_2wheels`
  - 两轮差速版 XLerobot 实现
- `examples/4_xlerobot_teleop_keyboard.py`
  - 原始 XLerobot 键盘遥操作
- `examples/4_xlerobot_2wheels_teleop_keyboard.py`
  - 两轮差速版键盘遥操作
- `examples/shield/4_xlerobot_2wheels_teleop_keyboard_safe_exit.py`
  - 带安全退出流程的两轮差速版键盘遥操作

## 安装

### 1. 环境要求

- Python `>=3.12`
- 建议使用虚拟环境或 conda 环境
- 如果控制 XLerobot 实机，需要 Feetech 电机依赖

### 2. 推荐安装方式

在仓库根目录执行：

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
source .venv/bin/activate
```

安装仓库本体和 XLerobot 常用依赖：

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[feetech]"
python -m pip install pyzmq
```

这里推荐额外安装 `pyzmq`，因为：

- `xlerobot_2wheels` 包会同时暴露本地类、host、client
- 即使你主要走直连模式，导入时也可能需要 ZMQ 依赖

### 3. 验证是否使用的是本地源码

```bash
python -c "import lerobot; print(lerobot.__file__)"
```

输出路径应该落在当前仓库的 `src/lerobot` 下，而不是旧的 `site-packages/lerobot`。

## 快速指南

这部分参考了 XLeRobot 官方软件文档里的“快速指南”思路，但改成了这个 fork 当前真实可运行的方式。

建议按下面顺序熟悉：

1. 先确保机械臂和电机总线通信正常
2. 再跑 XLerobot 键盘遥操作
3. 最后再进入更复杂的自定义开发

对于 XLerobot 基础版本或 2-wheel 版本，通常都可以直接使用你的笔记本电脑，不一定必须先上树莓派。

## XLerobot 的使用方式

### 1. 查串口

如果是直连实机，先查电机总线端口：

```bash
lerobot-find-port
```

对于这个仓库里的 `XLerobot2Wheels`，常见连接方式是：

- `port1`: 左臂 + 头部
- `port2`: 右臂 + 底盘

你需要把示例脚本中的端口改成你机器上的实际值。

### 2. 原始 XLerobot 键盘遥操作

运行：

```bash
python ./examples/4_xlerobot_teleop_keyboard.py
```

这个脚本对应：

- `src/lerobot/robots/xlerobot`

### 3. 两轮差速版 XLerobot 键盘遥操作

先修改：

- `examples/4_xlerobot_2wheels_teleop_keyboard.py`

把其中的：

- `robot id`
- `port1`
- `port2`

改成你自己的机器人配置，然后运行：

```bash
python ./examples/4_xlerobot_2wheels_teleop_keyboard.py
```

这个脚本对应：

- `src/lerobot/robots/xlerobot_2wheels`

### 4. Shield 安全退出版本

运行：

```bash
python ./examples/shield/4_xlerobot_2wheels_teleop_keyboard_safe_exit.py
```

这个版本适合两轮差速机器人，因为它在退出时做了受控处理：

- 可用 `V` 记录当前姿态为安全位
- 退出时先短暂停住当前姿态
- 再回到安全位
- 最后再断开

如果你担心原始脚本退出时手臂直接下坠，优先使用这个版本。

### 5. Host / Client 网络模式

如果电机连接在另一台电脑上，可以把那台电脑作为 host。

在连接机器人硬件的电脑上运行：

```bash
PYTHONPATH=src python -m lerobot.robots.xlerobot_2wheels.xlerobot_2wheels_host --robot.id=my_xlerobot_2wheels --robot.port1=COM5 --robot.port2=COM4
```

在客户端脚本中使用：

```python
from lerobot.robots.xlerobot_2wheels import XLerobot2WheelsClient, XLerobot2WheelsClientConfig

config = XLerobot2WheelsClientConfig(remote_ip="ROBOT_PC_IP", id="my_xlerobot_2wheels")
robot = XLerobot2WheelsClient(config)
robot.connect()
```

## 常用命令

查看安装信息：

```bash
lerobot-info
```

查找串口：

```bash
lerobot-find-port
```

查看训练帮助：

```bash
lerobot-train --help
```

查看评估帮助：

```bash
lerobot-eval --help
```

## 常见问题

### 1. 导入到了旧的 `site-packages/lerobot`

重新执行 editable install，并确认：

```bash
python -c "import lerobot; print(lerobot.__file__)"
```

### 2. `xlerobot_2wheels` 导入时报 `zmq` 相关错误

安装：

```bash
python -m pip install pyzmq
```

### 3. 电机扫描不到 ID

优先检查：

- 驱动是否正确
- 电机总线是否上电
- 串口是否接对

在 Windows / HarmonyOS PC 场景下，如果你使用的是 CH343 类串口芯片，可能需要正确的 WCH 驱动。

### 4. 退出时机械臂突然掉下去

使用：

- `examples/shield/4_xlerobot_2wheels_teleop_keyboard_safe_exit.py`

## 上游与参考

本仓库基于：

- 上游 LeRobot：<https://github.com/huggingface/lerobot>
- XLeRobot 软件文档：<https://xlerobot.readthedocs.io/zh-cn/latest/software/index.html>

这个 fork 的目标不是替代上游 LeRobot，而是把 XLerobot 接入 LeRobot 体系，让实机控制、数据采集和后续开发放在同一个代码仓里持续演进。
