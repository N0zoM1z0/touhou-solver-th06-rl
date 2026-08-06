#!/usr/bin/env python3

from th06_rl.th06.controller import main


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)

