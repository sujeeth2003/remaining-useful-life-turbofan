"""Data loading for run-to-failure engine data.

Two sources, same format (columns: unit, cycle, 3 operating settings, 21 sensors):
  * NASA C-MAPSS turbofan, e.g. train_FD001.txt  (download from the NASA Prognostics Data Repository; not redistributed here)
  * a synthetic generator with the same shape and qualitative behaviour, so the pipeline runs anywhere
"""
import numpy as np
import pandas as pd

SETTINGS = ["s1", "s2", "s3"]
SENSORS = [f"x{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle"] + SETTINGS + SENSORS


