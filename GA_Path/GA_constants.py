# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:50:05 2020

@author: jlwei
"""

import numpy as np

class GA_constants:
        
    def __init__(self):
        
        self.dimensions = 2   
        self.G = 6.67408e-11
        self.M_E = 5.9723e24
        self.R_E = 6378.137e3
        
        self.mass = 200
        self.thrust = 50000
        self.dt = 50.0
        self.alt_init = 300e3
        self.alt_final = 350e3
        
        self.acc = self.thrust/self.mass
        self.parameters = np.array([self.acc, self.dt])