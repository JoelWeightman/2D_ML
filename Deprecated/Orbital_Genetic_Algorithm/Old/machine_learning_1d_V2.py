# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 14:36:26 2018

@author: joellukw

A particle is accelerated at time t_0 at a value of acc_0, at t_1, the acceleration 
changes to acc_1. Cumulative sum is used for integration. No smoothing.

"""


import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.integrate as integ
    
def fitness(pop, weights, goals, thresh, t_steps, acc, v_max):

    pop['actions'][:,-1] = 0
    
    loc_val = np.abs(pop['results'][:,0] - goals[0]) / goals[0]
    vel_val = np.abs(pop['results'][:,1] - goals[1]) / v_max
    
    loc_val[loc_val < thresh[0]] = 0
    vel_val[vel_val < thresh[1]] = 0
    
    goal_val = np.concatenate((loc_val[:,np.newaxis],vel_val[:,np.newaxis]), axis = 1)
    pop['score'] = (weights[0] - weights[0]*goal_val[:,0]) + (weights[1] - weights[1]*goal_val[:,1])

    fuel_sum = np.cumsum(np.abs(pop['actions'][:,::-1]), axis = 1)[:,::-1]
    nonzero_length = np.argmin(fuel_sum, axis = 1)
    pop['score'] += (weights[2] - weights[2] * nonzero_length / t_steps) + (weights[3] - weights[3] * fuel_sum[:,0]/t_steps)
    pop['score'] *= 1/np.sum(weights)
    return pop

def init_population(pop):
    
    pop['actions'] = np.random.randint(-1, 2, size = np.shape(pop['actions']))
    
    return pop

def calculate_result(pop, acc, dt, best_index, final):
          
    if final == False:
                    
        v, d = velocity_distance(pop,dt,acc,t_steps)
            
        pop['results'] = np.concatenate((d,v), axis = 1)
    
        return pop
            
    else:
        
        v1 = np.zeros((np.shape(pop['actions'])[0],np.shape(pop['actions'])[1],1))
        
        v1[:,1:t_steps,0] = np.cumsum(pop['actions'][:,:-1]*dt*acc, axis = 1)
        d1 = np.cumsum(v1*dt, axis = 1)
            
        plt.figure()
        plt.scatter(np.arange(np.size(pop['actions'][0,:])),d1[best_index,:,0], s = 1, c='red')
        plt.figure()
        plt.scatter(np.arange(np.size(pop['actions'][0,:])),v1[best_index,:,0], s = 1,c='black')
        
        return
    
def velocity_distance(pop,dt,acc,t_steps):
    
    v = np.zeros((np.shape(pop['actions'])[0],np.shape(pop['actions'])[1],1))

    v[:,1:t_steps,0] = np.cumsum(np.multiply(pop['actions'][:,:-1],acc*dt), axis = 1)
    d = np.trapz(v, dx = dt, axis = 1)
    
    return v[:,-1], d
    
def pop_selection(pop, selection_num):
    
    sorted_pop = np.argsort(pop['score'])[::-1]
    
    elite_pop = sorted_pop[:selection_num[0]]
    selected_pop = sorted_pop[:selection_num[1]]
    lucky_pop = np.random.choice(sorted_pop[selection_num[1]:],size = selection_num[2], replace = False)
    mutated_pop = np.random.choice(selected_pop, size = selection_num[3], replace = False)
    selected_pop = np.setdiff1d(selected_pop,mutated_pop)
    
    actions = pop['actions'][np.concatenate((elite_pop,lucky_pop))]
    
    pop = generate_children(pop, actions, selection_num, selected_pop, mutated_pop)
    
    return pop, [np.array(pop['score'][sorted_pop[0]]),pop['results'][sorted_pop[0]],sorted_pop[0]]

def generate_children(pop, actions, selection_num, selected_pop, mutated_pop):
    
    mutated_actions = pop['actions'][mutated_pop,:]
    mut_num = np.random.randint(1,selection_num[4]+1,size = 1)[0]
    indices = np.random.randint(0,np.size(mutated_actions[0,:]), size = (np.size(mutated_actions[:,0]),mut_num))
    for i, mut_locs in enumerate(indices):
        mutated_actions[i,mut_locs] = np.random.randint(-1, 2, size = mut_num)
    
    random_selection_children_0 = np.random.randint(0,2,size = (int(len(selected_pop)/2),np.size(pop['actions'][0,:])))
    random_selection_children_1 = 1 - random_selection_children_0
    
    selected_pop_0 = np.random.choice(selected_pop, size = int(len(selected_pop)/2), replace = False)
    selected_pop_1 = np.setdiff1d(selected_pop, selected_pop_0)
    
    children_actions_0 = pop['actions'][selected_pop_0,:]*random_selection_children_0 + pop['actions'][selected_pop_1,:]*random_selection_children_1
    children_actions_1 = pop['actions'][selected_pop_1,:]*random_selection_children_0 + pop['actions'][selected_pop_0,:]*random_selection_children_1

    pop['actions'] = np.concatenate((actions,mutated_actions,children_actions_0,children_actions_1), axis = 0)

    return pop

if __name__ == "__main__":
    
    plt.close("all")
    ### Design Values
    loc_des = 100
    vel_des = 0#np.random.uniform(0,2)
    ideal_time_perc = 0.8#np.random.random()/2 + 0.5
    dt = 1.0
    
    goals = np.array([loc_des, vel_des, ideal_time_perc])
    
    ### Weightings of location, velocity, fuel used and time taken to reach dest
    A_l = 0.25
    A_v = 0.5
    A_f = 0
    A_t = 0.5
    
    weights = np.array([A_l, A_v, A_t, A_f])
    
    # Threshold Values
    thresh_loc = 0.0
    thresh_vel = 0.0
    thresh_time = 0.0
    thresh_perf = 1e-5
    
    thresh = np.array([thresh_loc, thresh_vel, thresh_time])
    
    ## Other Set Params
    elite_perc = 0.05
    luck_perc = 0.10
    mutation_perc = 0.40
    mutation_chance = 0.05
    
    ## Conditions to set
    times = [100]
    pops = [10000]
    samples = 1
    gen_count = np.zeros((len(pops),len(times),2,samples))
    for JJ, t_steps in enumerate(times):
        for II, pop_size in enumerate(pops):
            for KK in range(samples):
                start_time = time.time()
                generations = 1000
                 
                pop = dict({'actions': np.zeros((pop_size,t_steps)), 'results':np.zeros((pop_size,2)), 'score':np.zeros((pop_size))})
                

                
                ## Calculated Parameters
                pop_num_elite = int(elite_perc*pop_size)
                pop_num_mutation = int(mutation_perc*pop_size)
                pop_num_lucky = int((pop_size*luck_perc))
                total_non_children = pop_num_lucky + pop_num_elite + pop_num_mutation
                pop_num_selected = pop_size - total_non_children + pop_num_mutation
                if np.mod(pop_num_selected - pop_num_mutation,2) != 0:
                    pop_num_lucky += 1
                    pop_num_mutation -= 1
                    
                mutation_gene_num = int(mutation_chance*t_steps)
                selection_num = np.array([pop_num_elite, pop_num_selected, pop_num_lucky, pop_num_mutation, mutation_gene_num])
                
                v_max = loc_des/(dt*t_steps*ideal_time_perc/2)
                acc = v_max/(dt*t_steps*ideal_time_perc/2)
                theory_max = (A_l + A_v + A_t*(1-ideal_time_perc))/(A_l + A_v + A_t)
                
                COUNT = 0
                store_perf = 0
                ## First iteration
                pop = init_population(pop)
                ## Calculate Generations until convergence or theory max               
                for I in range(generations):
                    
                    pop = calculate_result(pop, acc, dt, [], False)
                    pop = fitness(pop, weights, goals, thresh, t_steps, acc, v_max)
                    pop, best_performance = pop_selection(pop, selection_num)
                    
                    if np.mod(I,10) == 0:
                        print('t: %d, Pop: %d, Run: %1d, Max Perf: %3.3f' % (t_steps,pop_size,KK, theory_max))
                        print('Gen %1d Performance %3.3f, Distance = %3.1f, Velocity = %3.3f' % (I, best_performance[0], best_performance[1][0], best_performance[1][1]))
                        
                    if best_performance[0] >= theory_max-thresh_perf:
                        COUNT += 1
                        if COUNT >= 2:
                            break
                elapsed_time = time.time() - start_time
                gen_count[II,JJ,0,KK] = I
                gen_count[II,JJ,1,KK] = elapsed_time

    gen_count_mean = np.mean(gen_count,3)
    calculate_result(pop, acc, dt, best_performance[2], True)
   
    plt.figure()
    plt.scatter(np.arange(len(pop['actions'][best_performance[2],:])),pop['actions'][best_performance[2],:],s=1,c='k')
    print('Gen %1d Performance %3.3f, Distance = %3.1f, Velocity = %3.3f' % (I, best_performance[0], best_performance[1][0], best_performance[1][1]))
    print('Ideal: Distance = %3.1f, Velocity = %3.3f, Time Percentage = %3.3f' % (loc_des, vel_des, ideal_time_perc))
    

    