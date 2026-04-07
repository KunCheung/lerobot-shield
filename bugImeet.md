# Bugs I Met

This file records the issues encountered while bringing up the XLERobot 2-wheels project, along with the corresponding fixes and verification results.

## 1. Python `lzma` import failure

### Symptom

Running the example script failed very early with:

```text
ImportError: DLL load failed while importing _lzma: The specified module could not be found.
```

### Root Cause

The Conda environment `C:\softwares\miniconda\envs\lerobot` had a broken `lzma` runtime. The Python extension `_lzma.pyd` existed, but the required `liblzma.dll` was missing from:

```text
C:\softwares\miniconda\envs\lerobot\Library\bin
```

### Fix

Restore `liblzma.dll` into the environment, for example from the local Conda package cache:

```powershell
Copy-Item `
  C:\softwares\miniconda\pkgs\xz-5.8.2-h53af0af_0\Library\bin\liblzma.dll `
  C:\softwares\miniconda\envs\lerobot\Library\bin\liblzma.dll `
  -Force
```

Then verify:

```powershell
C:\softwares\miniconda\envs\lerobot\python.exe -c "import lzma; print('lzma ok')"
```

### Result

`lzma` imported successfully again.

## 2. Missing `pyzmq`

### Symptom

Importing the XLERobot 2-wheels package could fail with:

```text
ModuleNotFoundError: No module named 'zmq'
```

### Root Cause

The environment did not have `pyzmq` installed, but the package imports the host/client ZMQ modules.

### Fix

Install `pyzmq` in the active environment:

```powershell
pip install pyzmq
```

### Result

The ZMQ-related modules could be imported normally.

## 3. Robot connection failed with `There is no status packet!`

### Symptom

Connecting to the robot failed with an error similar to:

```text
Failed to write 'Min_Position_Limit' on id_=7 with '1309' after 1 tries. [TxRxResult] There is no status packet!
```

### Root Cause

At first this looked like a bad motor ID or wrong COM port, but a direct scan showed that neither `COM4` nor `COM5` could see any Feetech motors from this HarmonyOS computer. The actual issue was the USB serial driver on this machine.

This computer was using the generic Microsoft driver:

```text
USB serial device
Manufacturer: Microsoft
```

while the working computer used:

```text
USB-Enhanced_SERIAL CH343
Provider: wch.cn
```

### Fix

Replace the generic Microsoft USB serial driver with the official WCH CH343 driver. The issue was fixed by following:

```text
https://blog.csdn.net/wch_techgroup/article/details/124801135
```

After installing the WCH driver, re-scan the ports:

```powershell
C:\softwares\miniconda\envs\lerobot\python.exe -c "import sys; sys.path.insert(0, r'c:\projects\lerobot\src'); from lerobot.motors.feetech import FeetechMotorsBus; print('COM4', FeetechMotorsBus.scan_port('COM4')); print('COM5', FeetechMotorsBus.scan_port('COM5'))"
```

### Result

The ports started returning the expected motor IDs:

```text
COM4 {1000000: [1, 2, 3, 4, 5, 6, 9, 10]}
COM5 {1000000: [1, 2, 3, 4, 5, 6, 7, 8]}
```

This matched the code expectation for:

- `COM4`: right arm + base
- `COM5`: left arm + head

## 4. `lerobot-find-cameras` saved pure white images

### Symptom

Running:

```powershell
lerobot-find-cameras
```

or:

```powershell
lerobot-find-cameras opencv
```

could detect cameras and still save blank white images in:

```text
outputs/captured_images
```

The saved images were not merely overexposed. They were fully white frames, for example:

```text
mean=255.0, std=0.0, min=255, max=255
```

### Root Cause

There were two overlapping issues:

1. On this Windows machine, some OpenCV capture paths could return invalid blank frames even when a camera was successfully enumerated.
2. After opening a new PowerShell window, running the bare command:

```powershell
lerobot-find-cameras
```

often invoked the previously installed console entrypoint instead of the modified local source in this repository. That meant the newer blank-frame rejection logic in:

[`src/lerobot/scripts/lerobot_find_cameras.py`](C:/projects/lerobot/src/lerobot/scripts/lerobot_find_cameras.py)

was not actually being used.

### Fix / Workaround

Use the repository source directly instead of relying on the old installed command:

```powershell
cd C:\projects\lerobot
conda activate lerobot
python -m lerobot.scripts.lerobot_find_cameras opencv --record-time-s 1
```

To make the console command use the local source more reliably, reinstall the project in editable mode:

```powershell
cd C:\projects\lerobot
conda activate lerobot
python -m pip install -e .
```

Then verify the active package path:

```powershell
python -c "import lerobot; print(lerobot.__file__)"
```

The output should point into:

```text
C:\projects\lerobot\src\lerobot
```

instead of an old `site-packages` copy.

### Result

The issue was identified as a command-entrypoint mismatch plus unstable Windows OpenCV capture behavior, not simply a bad camera.

The local source version of `lerobot_find_cameras.py` was updated to:

- reject blank white / black validation frames
- try multiple OpenCV backend / FOURCC combinations
- remove stale captured images before a new run
- print explicit capture statistics for each run

This made it much easier to tell whether a run produced fresh valid images or only reused stale white captures.

## 5. `move_to_you.py` sent a white RoboCrew navigation template to the LLM

### Symptom

When running:

```powershell
cd C:\projects\lerobot
conda activate lerobot
python examples/shield/move_to_you.py
```

the saved debug images showed:

- `latest_raw_camera.png` was a valid real camera frame
- `latest_llm_input.jpg` was a white or nearly white RoboCrew-style navigation image

The model output matched that bad input and often described a blank or white scene with a grid instead of the real camera view.

### Root Cause

The issue was not the validated local OpenCV camera itself. The real root cause was that the default RoboCrew camera path was not stably reusing the validated local capture stream. As a result, the image that eventually reached the LLM could diverge from the real camera frame and degrade into a blank navigation template.

### Fix

The camera path in [`examples/shield/move_to_you.py`](C:/projects/lerobot/examples/shield/move_to_you.py) was changed so that:

- the real camera frame is first captured from the validated local OpenCV stream
- that real frame is saved to `latest_raw_camera.*` for debugging
- RoboCrew's `basic_augmentation(...)` is applied directly on top of the real frame
- the LLM receives one final image: the real scene with RoboCrew's navigation grid overlaid on it

The script also saves:

- `examples/shield/tmp_images/latest_raw_camera.png`
- `examples/shield/tmp_images/latest_raw_camera.json`
- `examples/shield/tmp_images/latest_llm_input.jpg`
- `examples/shield/tmp_images/latest_llm_input.json`

### Result

`latest_llm_input.jpg` now represents the same real scene as `latest_raw_camera.png`, but with RoboCrew's grid overlay. The model input updates as the robot moves and retains both visual scene content and navigation guidance.

## 6. Calibration JSON failed to load because of UTF-8 BOM

### Symptom

Running the safe teleoperation script could fail immediately with:

```text
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

This happened when launching:

```powershell
python .\examples\shield\4_xlerobot_2wheels_teleop_keyboard_safe_exit.py
```

### Root Cause

The local calibration JSON had been rewritten by a Windows/PowerShell path and ended up with a UTF-8 BOM. The calibration loader used a normal `open()` call, so the first BOM byte sequence was treated as invalid JSON input.

### Fix

Two fixes were applied:

1. The local calibration file for `my_xlerobot_2wheels_lab` was rewritten as plain UTF-8 without BOM.
2. The calibration loading and saving helpers were hardened:

- [`src/lerobot/robots/robot.py`](C:/projects/lerobot/src/lerobot/robots/robot.py) now loads with `encoding="utf-8-sig"` and saves with `encoding="utf-8"`
- [`src/lerobot/teleoperators/teleoperator.py`](C:/projects/lerobot/src/lerobot/teleoperators/teleoperator.py) now loads with `encoding="utf-8-sig"` and saves with `encoding="utf-8"`

### Result

Calibration files now load correctly whether they contain a BOM or not, and future saves from the repository code path are written as plain UTF-8 without BOM.

## 7. `xlerobot_2wheels` base direction was wrong even with corrected key mapping

### Symptom

The safe teleoperation script showed the wrong physical mapping:

- `i` turned left
- `k` turned right
- `u` moved forward
- `o` moved backward

even though the printed keymap and configuration still said:

- `i`: forward
- `k`: backward
- `u`: rotate left
- `o`: rotate right

### Root Cause

There were two confirmed causes.

First, recalibration could overwrite the base wheel `drive_mode` back to `0`, which erased the left wheel inversion needed by this specific robot.

Second, and more importantly, the differential-drive base uses the raw velocity path:

- `Goal_Velocity`
- `Present_Velocity`

That path did not go through the existing position normalization logic, so `drive_mode` was never applied to base velocity commands or feedback. In practice, this meant that even with:

```text
base_left_wheel.drive_mode = 1
base_right_wheel.drive_mode = 0
```

the robot still moved as if both wheels were non-inverted.

### Fix

The following changes were made in [`src/lerobot/robots/xlerobot_2wheels/xlerobot_2wheels.py`](C:/projects/lerobot/src/lerobot/robots/xlerobot_2wheels/xlerobot_2wheels.py):

- `calibrate()` now preserves the existing base wheel `drive_mode` values instead of forcing them back to `0`
- the local calibration for `my_xlerobot_2wheels_lab` keeps:
  - `base_left_wheel.drive_mode = 1`
  - `base_right_wheel.drive_mode = 0`
- a dedicated base-velocity correction step now applies `drive_mode` to both:
  - outgoing `Goal_Velocity`
  - incoming `Present_Velocity`

The behavior was verified with:

```powershell
C:\softwares\miniconda\envs\lerobot\python.exe -m pytest tests\robots\test_xlerobot_2wheels_drive_mode.py -q
```

and then re-checked on hardware with:

```powershell
python .\examples\shield\4_xlerobot_2wheels_teleop_keyboard_safe_exit.py
```

### Result

The physical robot behavior finally matched the intended semantics of the keymap:

- `i` forward
- `k` backward
- `u` rotate left
- `o` rotate right
