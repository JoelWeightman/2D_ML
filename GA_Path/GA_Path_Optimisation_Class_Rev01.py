# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 22:55:36 2020

@author: JL
"""

import numpy as np
import Init_constants
import GA_algorithm
import GA_Path

        
def initialise_classes():
    
    pop = GA_Path.Path()
    constants = Init_constants.Constants()
    GA_values = GA_algorithm.GA_alg(constants.dimensions)
    
    return pop, constants, GA_values

def run_GA(pop, constants, GA_values):
      
    count = 0
    max_score = 0    

    # initialise actions
    pop.initialise_population(GA_values)
    pop.initialise_values(constants, GA_values)
    
    # run path calc and score pop
#    pop.result = pop.calculate_path(pop.actions,constants)
    pop.result = pop.calculate_path_odeint(pop.actions,constants)
    pop.score = GA_values.score_population(pop,pop.result)
    
    for i in range(GA_values.max_gen):
        print('Generation: ',i)
        #generate potential children and test
        pop.children_actions = GA_values.generate_children(pop,constants.dimensions)
#        pop.children_result = pop.calculate_path(pop.children_actions,constants)
        pop.children_result = pop.calculate_path_odeint(pop.children_actions,constants)
        pop.children_score = GA_values.score_population(pop,pop.children_result)
        
        # Official next gen
        pop = GA_values.generate_next_generation(pop)
        
        if np.max(pop.score) == max_score:
            count += 1
            print('No Progress')
        else:
            count = 0
            max_score = np.max(pop.score)
            print(max_score)
        
        if count == GA_values.max_plateau:
            print('GA is stuck in a plateau')
            break
    
    return pop
    
if __name__ == "__main__":
    
    pop, constants, GA_values = initialise_classes()
    
    pop.run_initialisation(constants)
    pop = run_GA(pop, constants, GA_values)

#    pop.init_conditions = pop.get_init_conditions(constants)
#    pop.final_conditions = pop.get_final_conditions(constants)
#    
#    

    

    