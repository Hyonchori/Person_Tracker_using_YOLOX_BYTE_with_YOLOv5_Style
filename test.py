import os
import sys
from pathlib import Path

FILE = Path(__file__).absolute()
if os.path.join(FILE.parents[0], "yolox") not in sys.path:
    sys.path.append(os.path.join(FILE.parents[0], "yolox"))
from yolox.exp import get_exp as get_yolox_exp

