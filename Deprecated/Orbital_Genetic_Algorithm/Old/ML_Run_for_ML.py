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
from multiprocessing import Pool



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
                    
        v, d = velocity_distance(pop,dt,acc)
            
        pop['results'] = np.concatenate((d,v), axis = 1)
    
        return pop
            
    else:
        
        v1 = np.zeros((np.shape(pop['actions'])[0],np.shape(pop['actions'])[1],1))
        
        v1[:,1:np.shape(pop['actions'])[1],0] = np.cumsum(pop['actions'][:,:-1]*dt*acc, axis = 1)
        d1 = np.cumsum(v1*dt, axis = 1)
            
        plt.figure()
        plt.scatter(np.arange(np.size(pop['actions'][0,:])),d1[best_index,:,0], s = 1, c='red')
        plt.figure()
        plt.scatter(np.arange(np.size(pop['actions'][0,:])),v1[best_index,:,0], s = 1,c='black')
        
        return
    
def velocity_distance(pop,dt,acc):
    
    v = np.zeros((np.shape(pop['actions'])[0],np.shape(pop['actions'])[1],1))

    v[:,1:np.shape(pop['actions'])[1],0] = np.cumsum(np.multiply(pop['actions'][:,:-1],acc*dt), axis = 1)
    d = np.trapz(v, dx = dt, axis = 1)
    
    return v[:,-1], d
    
def pop_selection(pop, selection_num):
    
    sorted_pop = np.argsort(pop['score'])[::-1]
    
    elite_pop = sorted_pop[:selection_num[0]]
    selected_pop = sorted_pop[:selection_num[1]]
    lucky_pop = np.random.choice(sorted_pop[selection_num[1]:], size = selection_num[2], replace = False)
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

def run(A_l, A_v, A_f, A_t, perc_elite, perc_lucky, perc_mutation, perc_selected, mutation_chance, pop_size, generations = 10000, t_steps = 200):
    
    perc_elite, perc_lucky, perc_mutation, perc_selected = [perc_elite, perc_lucky, perc_mutation, perc_selected]/(perc_elite + perc_lucky + perc_mutation + perc_selected)
    A_f = 0
    plt.close("all")
    ### Design Values
    loc_des = 100
    vel_des = 0#np.random.uniform(0,2)
    ideal_time_perc = 0.8#np.random.random()/2 + 0.5
    dt = 1.0
    max_pop = 2000
    
    pop_size = int(np.round(pop_size*max_pop))
    if pop_size < 20:
        pop_size = 20
    
    # Threshold Values
    thresh_loc = 0.0
    thresh_vel = 0.0
    thresh_time = 0.0
    thresh_perf = 1e-5
    
    
    thresh = np.array([thresh_loc, thresh_vel, thresh_time])
    weights = np.array([A_l, A_v, A_t, A_f])
    goals = np.array([loc_des, vel_des, ideal_time_perc])
    
    ## Conditions to set
    samples = 20
    gen_count = np.zeros((2,samples))
    
    for KK in range(samples):
        start_time = time.time()
        print(KK)
        pop = dict({'actions': np.zeros((pop_size,t_steps)), 'results':np.zeros((pop_size,2)), 'score':np.zeros((pop_size))})
        
        ## Calculated Parameters
        pop_num_elite = int(perc_elite*pop_size)
        pop_num_mutation = int(perc_mutation*pop_size)
        pop_num_lucky = int((perc_lucky*pop_size))
        total_non_children = pop_num_lucky + pop_num_elite + pop_num_mutation
        pop_num_selected = pop_size - total_non_children + pop_num_mutation
        if np.mod(pop_num_selected - pop_num_mutation,2) != 0:
            pop_num_selected += 1
            pop_num_elite -= 1
            
        mutation_gene_num = int(mutation_chance*t_steps)
        selection_num = np.array([pop_num_elite, pop_num_selected, pop_num_lucky, pop_num_mutation, mutation_gene_num])
        
        v_max = loc_des/(dt*t_steps*ideal_time_perc/2)
        acc = v_max/(dt*t_steps*ideal_time_perc/2)
        theory_max = (A_l + A_v + A_t*(1-ideal_time_perc))/(A_l + A_v + A_t)
        
        COUNT = 0
        ## First iteration
        pop = init_population(pop)
        ## Calculate Generations until convergence or theory max               
        for I in range(generations):
            
            pop = calculate_result(pop, acc, dt, [], False)
            pop = fitness(pop, weights, goals, thresh, t_steps, acc, v_max)
            pop, best_performance = pop_selection(pop, selection_num)
            
#            if np.mod(I,1000) == 0:
#                print('t: %d, Pop: %d, Run: %1d, Max Perf: %3.3f' % (t_steps,pop_size,KK, theory_max))
#                print('Gen %1d Performance %3.3f, Distance = %3.1f, Velocity = %3.3f' % (I, best_performance[0], best_performance[1][0], best_performance[1][1]))
                
            if best_performance[0] >= theory_max-thresh_perf:
                COUNT += 1
                print('Gen %1d Performance %3.3f, Distance = %3.1f, Velocity = %3.3f' % (I, best_performance[0], best_performance[1][0], best_performance[1][1]))
                if COUNT >= 2:
                    break
        elapsed_time = time.time() - start_time
        gen_count[0,KK] = I
        gen_count[1,KK] = elapsed_time

    gen_count_stats = np.zeros((3))
    gen_count_stats[:2] = np.mean(gen_count,1)
    gen_count_stats[2] = np.sqrt(np.var(gen_count[1,:]))
    
    return gen_count_stats, pop, best_performance, acc, dt

def fitness_ML(pop_ML):
    
    pop_ML['score'] = np.exp(-(pop_ML['results'][:,1]/10))
    
    return pop_ML

def init_population_ML(pop_ML):
    
    pop_ML['actions'] = np.random.uniform(0.10,0.90,size = (np.shape(pop_ML['actions'])[0],np.shape(pop_ML['actions'])[1]))
    
    return pop_ML

def calculate_ML(pop_ML, current_gen):
    
    start_time = time.time()
    gen_stats = np.zeros((np.size(pop_ML['actions'],0),3))
    pool = Pool(10)
    
    A = []
    for i in range(np.size(pop_ML['actions'],0)):
        A.append((pop_ML['actions'][i,0],pop_ML['actions'][i,1],pop_ML['actions'][i,2],pop_ML['actions'][i,3],pop_ML['actions'][i,4],pop_ML['actions'][i,5],pop_ML['actions'][i,6],pop_ML['actions'][i,7],pop_ML['actions'][i,8],pop_ML['actions'][i,9]))
    print('Gen = %d' %(current_gen))
    for i, result in enumerate(pool.starmap(run, A)):
        print(current_gen, i)
#        result = pool.starmap(run, [(pop_ML['actions'][i,0],pop_ML['actions'][i,1],pop_ML['actions'][i,2],pop_ML['actions'][i,3],pop_ML['actions'][i,4],pop_ML['actions'][i,5],pop_ML['actions'][i,6],pop_ML['actions'][i,7],pop_ML['actions'][i,8])])
        gen_stats[i,:] = result[0]
        
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    
#    start_time = time.time()
#    for i in [0]:#range(np.size(pop_ML['actions'],0)):
#        print(current_gen, i)
#        result, pop, best_performance, acc, dt = run(pop_ML['actions'][i,0],pop_ML['actions'][i,1],pop_ML['actions'][i,2],pop_ML['actions'][i,3],pop_ML['actions'][i,4],pop_ML['actions'][i,5],pop_ML['actions'][i,6],pop_ML['actions'][i,7],pop_ML['actions'][i,8])
#        gen_stats[i,:] = result
#    elapsed_time = time.time() - start_time
#    print(elapsed_time)
    pop_ML['results'] = gen_stats
    pool.close()
    
    return pop_ML

def pop_selection_ML(pop_ML, selection_num_ML):
    
    sorted_pop = np.argsort(pop_ML['score'])[::-1]
    
    elite_pop = sorted_pop[:selection_num_ML[0]]
    selected_pop = sorted_pop[:selection_num_ML[1]]
    if len(selected_pop) == np.shape(pop_ML['actions'])[0]:
        lucky_pop = []
    else:
        lucky_pop = np.random.choice(sorted_pop[selection_num_ML[1]:],size = selection_num_ML[2], replace = False)
    mutated_pop = np.random.choice(selected_pop, size = selection_num_ML[3], replace = False)
    selected_pop = np.setdiff1d(selected_pop,mutated_pop)
    lucky_elite = np.concatenate((elite_pop,lucky_pop))
    if len(lucky_elite) == 0:
        actions = []
    else:
        actions = pop_ML['actions'][lucky_elite]
    
    pop_ML = generate_children_ML(pop_ML, actions, selection_num_ML, selected_pop, mutated_pop)
    
    return pop_ML, [np.array(pop_ML['score'][sorted_pop[0]]),pop_ML['results'][sorted_pop[0]],sorted_pop[0]]

def generate_children_ML(pop_ML, actions, selection_num_ML, selected_pop, mutated_pop):
    
    mutated_actions = pop_ML['actions'][mutated_pop,:]
    mut_num = np.random.randint(1,selection_num_ML[4]+1,size = 1)[0]
    indices = np.random.randint(0,np.size(mutated_actions[0,:]), size = (np.size(mutated_actions[:,0]),mut_num))
    for i, mut_locs in enumerate(indices):
        mutated_actions[i,mut_locs] = np.random.uniform(0.10, 0.9, size = mut_num)
    
    random_selection_children_0 = np.random.randint(0,2,size = (int(len(selected_pop)/2),np.size(pop_ML['actions'][0,:])))
    random_selection_children_1 = 1 - random_selection_children_0
    
    selected_pop_0 = np.random.choice(selected_pop, size = int(len(selected_pop)/2), replace = False)
    selected_pop_1 = np.setdiff1d(selected_pop, selected_pop_0)
    
    children_actions_0 = pop_ML['actions'][selected_pop_0,:]*random_selection_children_0 + pop_ML['actions'][selected_pop_1,:]*random_selection_children_1
    children_actions_1 = pop_ML['actions'][selected_pop_1,:]*random_selection_children_0 + pop_ML['actions'][selected_pop_0,:]*random_selection_children_1

    pop_ML['actions'] = np.concatenate((actions,mutated_actions,children_actions_0,children_actions_1), axis = 0)

    return pop_ML

def run_ML(generations, already_started = 0, pop_ML = None):

    pop_size = 20
    t_steps = 10
    
    perc_elite = 0.10
    perc_lucky = 0.0
    perc_mutation = 0.50
#    mutation_chance = 0.05
    
    pop_num_elite = int(perc_elite*pop_size)
    pop_num_mutation = int(perc_mutation*pop_size)
    pop_num_lucky = int((pop_size*perc_lucky))
    total_non_children = pop_num_lucky + pop_num_elite + pop_num_mutation
    pop_num_selected = pop_size - total_non_children + pop_num_mutation
    
    if np.mod(pop_num_selected - pop_num_mutation,2) != 0:
        pop_num_elite += 1
        pop_num_selected -= 1
            
    mutation_gene_num = 1#int(mutation_chance*t_steps)
    selection_num_ML = np.array([pop_num_elite, pop_num_selected, pop_num_lucky, pop_num_mutation, mutation_gene_num])
    
    time_to_locvel_ratio = 0.75
    
    if already_started == 0:
        print('Starting')
        pop_ML = dict({'actions': np.zeros((pop_size,t_steps)), 'results':np.zeros((pop_size,3)), 'score':np.zeros((pop_size))})
        pop_ML = init_population_ML(pop_ML)      
    else:
        print('Already Started, but Continuing')
    
    for I in range(generations):
        
        pop_ML['actions'][:,0] = 0.5*pop_ML['actions'][:,1]
        
        vel_loc_weight = (pop_ML['actions'][:,0] + pop_ML['actions'][:,1])
        bad_time_weights = pop_ML['actions'][:,3] > vel_loc_weight*time_to_locvel_ratio
        pop_ML['actions'][bad_time_weights,3] = (pop_ML['actions'][bad_time_weights,0] + pop_ML['actions'][bad_time_weights,1])*time_to_locvel_ratio

        pop_ML = calculate_ML(pop_ML, I)
        pop_ML = fitness_ML(pop_ML)
        pop_ML, best_performance = pop_selection_ML(pop_ML, selection_num_ML)
        
        print('BEST PERFORMANCE AT GEN %d: %3.3f' %(I, best_performance[0]))
        
        filename = 'Population_ML_Gen_Temp.npy'
        np.save(filename, pop_ML)
       
    return pop_ML, best_performance

if __name__ == "__main__":
       
    generations = 20
    
    already_started = 1
    
    if already_started == 0:
        pop_ML, best_performance = run_ML(generations, already_started, [])
    else:
        filename = 'Population_ML_Gen_Final.npy'
        pop_ML = np.load(filename).item()
        pop_ML, best_performance = run_ML(generations, already_started, pop_ML)
    
    i = best_performance[2]

    A_l, A_v, A_f, A_t, perc_elite, perc_lucky, perc_mutation, perc_selected, mutation_chance, pop_size = pop_ML['actions'][i,0],pop_ML['actions'][i,1],pop_ML['actions'][i,2],pop_ML['actions'][i,3],pop_ML['actions'][i,4],pop_ML['actions'][i,5],pop_ML['actions'][i,6],pop_ML['actions'][i,7],pop_ML['actions'][i,8],pop_ML['actions'][i,9]
    perc_elite, perc_lucky, perc_mutation, perc_selected = [perc_elite, perc_lucky, perc_mutation, perc_selected]/(perc_elite + perc_lucky + perc_mutation + perc_selected)
    pop_size = int(pop_size*2000)
    
    filename = 'Population_ML_Gen_Final.npy'
    np.save(filename, pop_ML)
    
    
    