# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 22:55:36 2020

@author: JL
"""

import numpy as np


class Constants:
        
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
    
class Path:
    """The class defines the path object for motion""" 
    

    def get_init_conditions(self,constants):
        
        R_E = constants.R_E
        G = constants.G
        M_E = constants.M_E
        dimensions = constants.dimensions
        alt_init = constants.alt_init
    
        alt = R_E + alt_init
        v = np.sqrt(G*M_E/(alt))
        
        if dimensions == 1:
            r_init = 0
            theta_init = 0
            phi_init = 90
            r_dot_init = 0
            theta_dot_init = 0
            phi_dot_init = 0
        elif dimensions == 2:
            r_init = alt
            theta_init = 0
            phi_init = 90
            r_dot_init = 0
            theta_dot_init = v/r_init
            phi_dot_init = 0
        elif dimensions == 3:
            r_init = alt
            theta_init = 0
            phi_init = 90
            r_dot_init = 0
            theta_dot_init = v/r_init
            phi_dot_init = 0
            
        return np.array([r_init, r_dot_init, np.deg2rad(theta_init), (theta_dot_init), 
                         np.deg2rad(phi_init), (phi_dot_init)])

    def get_final_conditions(self,constants):
    
        R_E = constants.R_E
        G = constants.G
        M_E = constants.M_E
        dimensions = constants.dimensions
        alt_final = constants.alt_final
        
        alt = R_E + alt_final
        v = np.sqrt(G*M_E/(alt))
        
        if dimensions == 1:
            r_final = 100
            theta_final = 0
            phi_final = 0
            r_dot_final = 0
            theta_dot_final = 0
            phi_dot_final = 0
        elif dimensions == 2:
            r_final = alt
            theta_final = 0
            phi_final = 90
            r_dot_final = 0
            theta_dot_final = v/r_final
            phi_dot_final = 0
        elif dimensions == 3:
            r_final = alt
            theta_final = 0
            phi_final = 90
            r_dot_final = 0
            theta_dot_final = v/r_final
            phi_dot_final = 0
        
        return np.array([r_final, r_dot_final, np.deg2rad(theta_final), (theta_dot_final), 
                     np.deg2rad(phi_final), (phi_dot_final)])

    def initialise_population(self,constants,genetic_alg):
        
        actions = np.random.rand(constants.dimensions,genetic_alg.t_steps)
        
        return actions
    
    
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
       
        
def initialise_classes():
    
    pop = Path()
    constants = Constants()
    genetic_alg = GenAlg()
    
    return pop, constants, genetic_alg

def run_GA(pop,constants,genetic_alg):
    
    pop.actions = pop.initialise_population(constants,genetic_alg)
    
    return pop
    
if __name__ == "__main__":
    
    pop,constants,genetic_alg = initialise_classes()

    pop.init_conditions = pop.get_init_conditions(constants)
    pop.final_conditions = pop.get_final_conditions(constants)
    
    run_GA

    

    