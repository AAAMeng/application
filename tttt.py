#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   tttt.py    
@Contact :   zhumeng@bupt.edu.cn
@License :   (C)Copyright 2019 zhumeng

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/11/28 17:26      xm         1.0          None
"""

import time
import numpy as np
import pandas as pd

num_list = [1, 1, 1, 1, 1]
num_list1 = [2, 2, 2, 2, 2]
error_list = []
error_list.append(num_list)
error_list.append(num_list1)
print(error_list)
f = pd.DataFrame(error_list)
print(f)