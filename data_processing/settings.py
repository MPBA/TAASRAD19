# -*- coding: utf-8 -*-

import datetime

import numpy as np
import pandas as pd

"""
Data generation settings. For additional info on how these are used take a look
at the `02_extract_sequences.py` script.

GAP_LENGTH : maximum time interval between two radar scans. If two scans have a temporal
different greater than GAP_LENGTH they do not belong to the same sequence.

MIN_RUN_SIZE : Minimum scan number for a sequences to not be discarded. 

IMG_SIZE : Input image size.

CROP_SIZE : Output image size.

SKIP_ROWS : Number of lines to skip at the beginning of every .ascii file

START_DATE : data start date.

END_DATE : Data end data.

PARSE_OFFSET : Offset used to crop the center region of the radar scan.
               PARSE_OFFSET = (IMG_SIZE - CROP_SIZE) / 2
               To crop a square inside the radar circle set
               CROP_SIZE = 340 (PARSE_OFFSET = 70)

PARSED_TYPE : datatype for HDF5 data
"""

GAP_LENGTH = pd.Timedelta("5 minutes")
MIN_RUN_SIZE = 25
IMG_SIZE = 480
CROP_SIZE = 480
SKIP_ROWS = 6
START_DATE = datetime.date(year=2010, month=6, day=1)
END_DATE = datetime.date(year=2020, month=12, day=31)
PARSE_OFFSET = (IMG_SIZE - CROP_SIZE) // 2
PARSED_TYPE = np.float16

########################### MASK GENERATION SETTINGS

# Radius of the circle of the scan. It is centered in the center of the scan.
RADIUS = IMG_SIZE / 2

# compute a circular mask inside a IMG_SIZE x IMG_SIZE square.
# In the scans are cropped, then the mask is cropped too
def _get_mask():
    X, Y = np.ogrid[:IMG_SIZE, :IMG_SIZE]
    mask = (
        np.sqrt(np.square(X - (IMG_SIZE-1) / 2) + np.square(Y - (IMG_SIZE-1) / 2)) <= RADIUS
    )
    if PARSE_OFFSET:
        return mask[PARSE_OFFSET:-PARSE_OFFSET, PARSE_OFFSET:-PARSE_OFFSET]
    else:
        return mask

MASK = _get_mask()

# Datatype to read scans in. Dependant on _data_converter below.
# Smaller datatypes means less memory usage/faster reads.
DATA_TYPE = np.uint16

# Minimum and maximum values for scans, counting pixels inside the circle.
SCAN_MIN_VALUE = 0
SCAN_MAX_VALUE = 52.5

# As inside the circle the values go from 0 to 52.5 with steps of 0.1, the total
# number of values is 526.
DISTINCT_VALUES = min(526, np.iinfo(DATA_TYPE).max + 1)
