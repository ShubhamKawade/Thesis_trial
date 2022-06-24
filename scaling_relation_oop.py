#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 09:46:05 2022

@author: schubham
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import bces.bces
from scipy import stats
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import general_functions as gf


class Scaling_relation:
    
    import numpy as np
    import bces.bces
    from scipy import stats 
    def __init__(self, x_data, norm_x, x_err, y_data, norm_y, y_err, gamma ):
        self.x = x_data
        self.y = y_data
        self.x_err = x_err
        self.yerr = y_err
        self.norm_x = norm_x
        self.norm_y = norm_y
        self.gamma = gamma
        
    
    
    def bestfit(self):
         a, b, a_err, b_err, cov_ab = bces.bces.bces(self.x_data, self.x_err, self.y_data, self.y_err, cov=0)
         scatter = gf.calculate_sig_intr_yx(self.x_data, self.y_data, self.x_err, self.y_err, a[0], b[0])

         return b[0], 10 ** b[0], a[0], scatter





