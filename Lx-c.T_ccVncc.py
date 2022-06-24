#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 13:13:55 2021

@author: schubham
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import general_functions
import seaborn as sns
cluster_total = pd.read_csv('/home/schubham/Thesis/Thesis/Data/cluster_total.csv')
cluster_total = general_functions.cleanup(cluster_total)

g = cluster_total.groupby('label')
CC_clusters = g.get_group('CC')
NCC_clusters = g.get_group('NCC')

