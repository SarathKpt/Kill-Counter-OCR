import os

os.environ["FENIKZ_OCR_DEVICE"] = "cpu"

from tray_app import main

if __name__ == "__main__":
    raise SystemExit(main())
