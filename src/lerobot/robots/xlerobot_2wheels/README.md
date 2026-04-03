# XLerobot 2-Wheels

`XLerobot2Wheels` 是这个仓库里对两轮差速版 XLerobot 的接入实现。

它运行在 LeRobot 的机器人接口之上，所以你可以继续复用 `lerobot` 的配置和工具链，同时直接控制两轮差速底盘、双臂和头部。

## 这个版本和原始 XLerobot 的区别

- 底盘从 3 轮全向改为 2 轮差速
- 主要支持：
  - 前进 / 后退
  - 原地旋转
- 不支持原始全向底盘那种横移

## 依赖安装

建议在仓库根目录安装：

```bash
python -m pip install -e ".[feetech]"
python -m pip install pyzmq
```

## 主要入口

- 标准键盘遥操作：
  - `examples/4_xlerobot_2wheels_teleop_keyboard.py`
- 安全退出版本：
  - `examples/shield/4_xlerobot_2wheels_teleop_keyboard_safe_exit.py`
- Host：
  - `lerobot.robots.xlerobot_2wheels.xlerobot_2wheels_host`

## 直连使用

先根据你的实际硬件修改脚本中的：

- `robot id`
- `port1`
- `port2`

然后运行：

```bash
python ./examples/4_xlerobot_2wheels_teleop_keyboard.py
```

如果你想使用带安全退出流程的版本：

```bash
python ./examples/shield/4_xlerobot_2wheels_teleop_keyboard_safe_exit.py
```

## Host / Client 模式

在连接机器人硬件的电脑上运行 host：

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

## 常见控制键

- `i` / `k`
  - 前进 / 后退
- `u` / `o`
  - 左旋 / 右旋
- `n` / `m`
  - 加速 / 减速

完整键位请以示例脚本启动后的终端输出为准。
