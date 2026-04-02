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


## 3. Missing `pyzmq`

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

## 4. Robot connection failed with `There is no status packet!`

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
