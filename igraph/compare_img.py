#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   compare_img.py    
@Contact :   zhumeng@bupt.edu.cn
@License :   (C)Copyright 2019 zhumeng

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/12/5 10:38      xm         1.0          None
"""

import tensorflow as tf
import numpy as np
import time
import os
from iclass.dataSet import DataSet
from sklearn.model_selection import train_test_split
import pandas as pd

tf.image.grayscale_to_rgb()