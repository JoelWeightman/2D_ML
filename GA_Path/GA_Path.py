# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:51:23 2020

@author: jlwei
"""

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
    
    