from __future__ import annotations

import sys

import common


if __name__ == "__main__":
    try:
        common.main()
    except KeyboardInterrupt:
        print("[Exit] Interrupted by user.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        raise SystemExit(1)
