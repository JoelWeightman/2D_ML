# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:50:44 2020

@author: jlwei
"""

import numpy as np
  
class GenAlg:
    
    def __init__(self):
        
        self.t_steps = 10
        self.pop_size = 50
        self.max_gen = 100
        self.samples = 1
        
        self.deg_steps = 360
        self.vel_steps = 1
          
        self.r_weight = 0.1
        self.r_vel_weight = 0.1
        self.theta_weight = 0.1
        self.theta_vel_weight = 0.1
        self.phi_weight = 0.1
        self.phi_vel_weight = 0.1
        self.t_weight = 0.1
            
        self.elite_weight = 0.1
        self.lucky_weight = 0.1
        self.children_weight = 0.1
        self.mutation_weight = 0.1
        self.mutation_chance = 0.1
       