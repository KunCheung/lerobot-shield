# Target Person Follow

这套示例位于 `examples/shield/move_to_target_person/`，用于两步完成“指定人物录入 + 指定人物靠近”：

先修正runtime_common.py 的default_camera， 使用机器人头部相机id，可通过lerobot-find-cameras命令查看./outputs/captured_images确定相机

1. 用当前摄像头为某个人录入参考图。
2. 运行跟随脚本，只识别并靠近该人物。


## 目录结构

```text
examples/shield/move_to_target_person/
├── capture_target_person_reference.py
├── move_to_target_person_by_cv.py
├── runtime_common.py
├── target_face_matcher.py
├── people_detection.py
├── follow_logic.py
├── reference_person/
│   ├── archive/
│   └── <person_name>/
├── models/
└── debug/
```

主要文件职责：

- `capture_target_person_reference.py`
  录入目标人物参考图。
- `move_to_target_person_by_cv.py`
  跟随入口脚本，负责运行时编排。
- `target_face_matcher.py`
  人脸检测、人脸特征提取、strict/fuzzy 身份匹配。
- `people_detection.py`
  生成人体候选框，内部使用 `PeopleDetectorStack`。
- `follow_logic.py`
  目标选择、跟踪状态和动作决策。
- `runtime_common.py`
  相机、机器人、调试输出、目录与输入校验等通用能力。

## 运行前准备

将下面两个 ONNX 模型放到 `models/` 目录：

- `face_detection_yunet_2023mar.onnx`
- `face_recognition_sface_2021dec.onnx`

目标人物图库按人名分文件夹组织：

```text
reference_person/alice/target_001.jpg
reference_person/alice/target_002.jpg
reference_person/bob/target_001.jpg
```

每次运行脚本时，都会先要求输入人物姓名。

## 1. 录入目标人物

运行：

```powershell
C:\softwares\miniconda\envs\lerobot\python.exe examples/shield/move_to_target_person/capture_target_person_reference.py
```

录入流程：

- 启动后先输入人物姓名。
- 若该人物已有旧图库，只会归档该人物目录到 `reference_person/archive/<person_name>/<timestamp>/`，不会影响其他人物。
- 只有在画面里恰好 1 张脸、脸足够大、清晰度足够时才允许保存。
- 默认最多采集 5 张，建议至少保留 3 张可用参考图。
- 按 `s` 保存当前帧。
- 按 `q` 或 `ESC` 退出。

建议采集方式：

- 正脸一张。
- 轻微左转一张。
- 轻微右转一张。
- 稍近一张。
- 稍远一张。

这样比只录入单一正脸更稳，后续匹配鲁棒性更好。

## 2. 跟随指定人物

默认 dry-run，只显示识别与动作决策，不驱动底盘：

```powershell
C:\softwares\miniconda\envs\lerobot\python.exe examples/shield/move_to_target_person/move_to_target_person_by_cv.py
```

如需允许机器人实际运动，显式加 `--live`：

```powershell
C:\softwares\miniconda\envs\lerobot\python.exe examples/shield/move_to_target_person/move_to_target_person_by_cv.py --live
```

跟随流程：

- 启动后先输入目标人物姓名。
- 只加载 `reference_person/<person_name>/` 下的参考图。
- 只对识别为该人物的目标执行转向和前进。
- 未识别到目标人物时，不会跟随普通路人。

当前运行链路是：

1. `PeopleDetectorStack` 生成人体候选框。
2. `TargetFaceMatcher` 做人脸身份匹配。
3. `follow_logic` 选择目标并生成动作。
4. `move_to_target_person_by_cv.py` 负责主循环、叠加显示、debug 输出和底盘执行。

目标丢失后的搜索策略：

- 如果上次能判断目标在画面左侧，则优先持续向左原地搜索。
- 如果上次能判断目标在画面右侧，则优先持续向右原地搜索。
- 如果没有上次方向信息，则默认固定向左开始搜索。
- 搜索会按固定步长累计到约 `360°`，而不是左右来回摆动。
- 一整圈仍未找回目标时，会退出当前搜索流程。

## 远距离小脸与 Fuzzy Match

这套实现支持远距离小脸的模糊匹配，用于机器人在 2-3 米外先开始靠近目标人物。

行为上有两档：

- `strict match`
  近距离或更清晰的人脸，匹配要求更严格。
- `fuzzy match`
  远距离小脸可走更宽松的匹配策略，但动作会更保守。

为了降低误跟风险，fuzzy 命中时会：

- 要求更多连续命中帧后才开始动作。
- 采用更小步长前进。

如果你后续要调远距离效果，优先查看：

- `target_face_matcher.py`
- `follow_logic.py`
- `move_to_target_person_by_cv.py`

## 调试输出

运行过程中，`debug/` 会保存最近一次调试文件：

- `last_capture_frame.jpg`
- `last_capture_face.jpg`
- `last_capture_status.txt`
- `last_frame.jpg`
- `last_target_face.jpg`
- `last_match.txt`

这些文件适合排查：

- 为什么录入时不能保存。
- 当前有没有识别到目标人物。
- 命中的是 strict 还是 fuzzy。

## 常见问题

### 1. 提示缺少模型文件

请确认 `models/` 目录下存在：

- `face_detection_yunet_2023mar.onnx`
- `face_recognition_sface_2021dec.onnx`

### 2. 输入的人名不存在

跟随脚本会报错，并列出当前可用的人物目录名。先确认 `reference_person/<person_name>/` 是否已经录入。

### 3. 录入时一直不能保存

常见原因：

- 画面里有多张脸。
- 脸太小。
- 脸太模糊。
- 光照太差，导致检测分数不够。

建议让目标人物单独出现在画面中，并先在中距离完成 3-5 张清晰录入。

### 4. 跟随时识别不到目标人物

优先检查：

- 模型文件是否齐全。
- 该人物目录下是否至少有 3 张质量合格的参考图。
- 目标人物当前是否露脸。
- 当前画面是否严重模糊或逆光。

### 5. 远距离能识别，但动作比较保守

这是预期行为。远距离小脸通常走 fuzzy match，系统会故意更谨慎，以降低误跟到其他人的概率。

### 6. 目标丢失后机器人一直左右摆动

当前版本已经改为“按上次方向优先的整圈搜索”：

- 有上次方向线索时，会持续朝该方向搜索。
- 没有线索时，会默认固定向左搜索。
- 搜索会覆盖约 `360°` 后再退出，不会无限左右交替。

## 说明

- README 以当前仓库内使用为目标，命令示例默认使用本机 `lerobot` 环境。
- 这份文档关注“怎么使用”和“怎么排查”，不展开所有内部阈值与算法细节。
