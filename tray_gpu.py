import os

os.environ["FENIKZ_OCR_DEVICE"] = "cuda"

from tray_app import main

if __name__ == "__main__":
    raise SystemExit(main())
