# SmolVLA Pick-and-Place

This folder is a standalone SmolVLA test area for the current shield robot.

## Dry Run

The dry-run path:

- load `Sa74ll/smolvla_so101_pickandplace`
- read the current right-arm joint state
- read `camera1` and `camera2`
- predict a 6D SO-101 right-arm action
- write debug files
- never execute predicted policy actions

From the repo root:

```powershell
conda activate lerobot
python .\examples\shield\smolvla_pickplace\run_dry.py
```

Useful options:

```powershell
python .\examples\shield\smolvla_pickplace\run_dry.py --help
python .\examples\shield\smolvla_pickplace\run_dry.py --max-steps 1 --no-window
python .\examples\shield\smolvla_pickplace\run_dry.py --camera camera1=1 --camera camera2=2
python .\examples\shield\smolvla_pickplace\run_dry.py --model-id C:\path\to\local\checkpoint
```

## Live Run

`run_live.py` is the conservative hardware entrypoint. It sends only the six right-arm position targets after clipping them against the current arm state.

First hardware test:

```powershell
conda activate lerobot
python .\examples\shield\smolvla_pickplace\run_live.py --confirm-live --max-steps 1 --step-confirm --max-joint-delta 3 --max-gripper-delta 3
```

Typical cached-chunk run:

```powershell
python .\examples\shield\smolvla_pickplace\run_live.py --confirm-live --use-cached-chunk --max-steps 120 --max-joint-delta 3 --max-gripper-delta 3 --settle-s 0.12
```

For the tissue checkpoint:

```powershell
python .\examples\shield\smolvla_pickplace\run_live.py --model-id Grigorij/smolvla_collect_tissues --task "Collect tissues." --confirm-live --use-cached-chunk --max-steps 120 --settle-s 0.12
```

Safety defaults:

- `--max-steps 500`
- `--max-joint-delta 3.0`
- `--max-gripper-delta 3.0`
- `--settle-s 0.20`
- `--step-confirm` is disabled by default; pass `--step-confirm` for one-command-at-a-time tests
- `--confirm-live` is required before any `send_action()` call
- the policy queue is reset before every live step unless `--use-cached-chunk` is passed

Use `--use-cached-chunk` only after single-step motion looks correct. On CPU, each fresh action chunk can take tens of seconds.

## Defaults

- model: `Sa74ll/smolvla_so101_pickandplace`
- task: `Pick and place the object.`
- robot id: `my_xlerobot_2wheels_lab`
- left arm + head port: `COM5`
- right arm + base port: `COM4`
- `camera1`: OpenCV index `1`
- `camera2`: OpenCV index `2`

## Debug Output

Dry-run creates:

```text
examples/shield/smolvla_pickplace/debug/<run_id>/
```

Live run creates:

```text
examples/shield/smolvla_pickplace/debug_live/<run_id>/
```

Key files:

- `session.json`: startup config and camera mapping
- `latest_preview.jpg`: latest two-camera preview with diagnostics
- `latest_runtime.txt`: latest timing, state, predicted action, and delta
- `latest_action.json`: latest action payload
- `actions.jsonl`: action payload history

Live action logs also include `clamped_action`, `clamped_delta`, `sent_action`, `was_clamped`, and camera freshness.

## Troubleshooting

If Hugging Face cannot be reached, download/cache the model first, then pass the local checkpoint path with `--model-id`.

If the script reports a checkpoint layout mismatch, the model is not a 6D SO-101 right-arm policy and should not be used with this dry-run.

If a camera opens as a blank frame, change the camera mapping with `--camera camera1=<index> --camera camera2=<index>`.

If OpenCV windows are unavailable, run with `--no-window`; debug images are still written.

If the first live step moves in the wrong direction, stop and check camera placement, task text, and whether `camera1`/`camera2` are swapped before increasing `--max-steps` or enabling cached chunks.
