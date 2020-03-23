# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 12:57:17 2019

@author: jlwei
"""

"""
Needed points:
    Set up goal
    Set up array sizes
    Set up initial guess
    Set up genetic algorithm
    Set up measurement of results
    Set up for loop to get correct result
    
"""

import numpy as np
import time
import scipy.optimize as optim
import random as rdm
import matplotlib.pyplot as plt


def get_acceleration(dimensions, goal, dt, t_steps, ideal_time_perc):
    
    [x_final,u_final,y_final,v_final] = goal
    
    if dimensions == 1:
        u_max = (x_final)/(dt*t_steps*ideal_time_perc/2)
        v_max = 0
        u_acc = u_max/(dt*t_steps*ideal_time_perc/2)
        v_acc = 0
        
    elif dimensions == 2:
        u_max = (x_final)/(dt*t_steps*ideal_time_perc/2)
        v_max = (y_final)/(dt*t_steps*ideal_time_perc/2)
        u_acc = u_max/(dt*t_steps*ideal_time_perc/2)
        v_acc = v_max/(dt*t_steps*ideal_time_perc/2)
    
    return [u_max, v_max], [u_acc, v_acc]          

def initialise_population(dimensions, t_steps, pop_size,steps):
    
    pop = dict({'actions': np.zeros((pop_size,t_steps,dimensions)), 'results':np.zeros((pop_size,dimensions*2)), 'score':np.zeros((pop_size))})

    pop['actions'] = initial_values(dimensions,pop['actions'],steps)

    return pop

def initial_values(dimensions, actions, steps):
#    steps = 4
    if dimensions == 1:
#        actions[:,:,0] = np.random.uniform(-1, 1, size = np.shape(actions[:,:,0]))
        actions[:,:,0] = np.random.randint(-steps, steps+1, size = np.shape(actions[:,:,0]))/steps
    elif dimensions == 2:
#        actions[:,:,0] = np.random.uniform(-1, 1, size = np.shape(actions[:,:,0]))
        actions[:,:,0] = np.random.randint(-steps, steps+1, size = np.shape(actions[:,:,0]))/steps
#        actions[:,:,1] = np.random.uniform(-1, 1, size = np.shape(actions[:,:,0]))
        actions[:,:,1] = np.random.randint(-steps, steps+1, size = np.shape(actions[:,:,0]))/steps
    
    return actions

def get_generation_nums(t_steps, pop_size, generation_weights):
    
    elite_num = np.round(generation_weights[0]*pop_size).astype('int64')
    lucky_num = np.round(generation_weights[1]*pop_size).astype('int64')
    children_num = (np.round(generation_weights[2]*pop_size/2)*2).astype('int64')
    non_mutated = [elite_num,lucky_num,children_num]
    if np.sum(non_mutated) > pop_size:
        non_mutated[np.argmax(non_mutated)] = np.max(non_mutated) - 1
        [elite_num,lucky_num,children_num] = non_mutated
    
    mutation_num = (pop_size - np.sum(non_mutated)).astype('int64')
    mutated_gene_max = np.round(generation_weights[-1]*100).astype('int64')
    
    generation_nums = np.array([elite_num,lucky_num,children_num,mutation_num,mutated_gene_max])
    
    return generation_nums

def population_results(dimensions, pop, dt, acc, t_steps):
    
    if dimensions == 1:
                    
        v, d = velocity_distance_1d(pop,dt,acc,t_steps)  
        pop['results'] = np.concatenate((d,v), axis = 1)
       
    elif dimensions == 2:
    
        u, v, d_x, d_y = velocity_distance_2d(pop,dt,acc,t_steps)
            
        pop['results'] = np.concatenate((d_x, u, d_y, v), axis = 1)

    return pop

def velocity_distance_1d(pop, dt, acc, t_steps):
    
    v = np.zeros((np.shape(pop['actions'])[0],np.shape(pop['actions'])[1],1))

    v[:,1:t_steps,0] = np.cumsum(np.multiply(pop['actions'][:,:-1],acc[0]*dt), axis = 1)[:,:,0]
    d = np.trapz(v, dx = dt, axis = 1)
    
    return v[:,-1], d
    

def velocity_distance_2d(pop, dt, acc, t_steps):
    
    v_x = np.zeros((np.shape(pop['actions'])[0],np.shape(pop['actions'])[1],1))
    v_y = np.zeros((np.shape(pop['actions'])[0],np.shape(pop['actions'])[1],1))

    v_x[:,1:t_steps,0] = np.cumsum(np.multiply(pop['actions'][:,:-1,0],acc[0]*dt), axis = 1)
    v_y[:,1:t_steps,0] = np.cumsum(np.multiply(pop['actions'][:,:-1,1],acc[1]*dt), axis = 1)

    d_x = np.trapz(v_x, dx = dt, axis = 1)
    d_y = np.trapz(v_y, dx = dt, axis = 1)
    
    return v_x[:,-1], v_y[:,-1], d_x, d_y

def population_score(dimensions, pop, result_weights, goal, parameters):

    [vel_max, acc, dt, ideal_time_perc] = parameters
    
    pop['actions'][:,-1,:] = 0.0
    
    loc_val = np.zeros((np.shape(pop['results'])[0]))
    vel_val = np.zeros((np.shape(pop['results'])[0]))
    goal_loc_val = np.zeros((np.shape(pop['results'])[0]))
    goal_vel_val = np.zeros((np.shape(pop['results'])[0]))
    
    for dim in np.arange(0,dimensions+1,2):
        
        loc_val += (pop['results'][:,dim] - goal[dim])**2
        goal_loc_val += goal[dim]**2
        
        vel_val += np.abs(pop['results'][:,dim+1] - goal[dim+1])
        
    for dim in range(dimensions):
        goal_vel_val +=  (vel_max[dim])
 
    loc_val = np.sqrt(loc_val) / np.sqrt(goal_loc_val)
    vel_val = (vel_val) / (goal_vel_val)

    pop['score'] = (result_weights[0] - result_weights[0]*loc_val) + (result_weights[1] - result_weights[1]*vel_val)
    
    fuel_sum = np.zeros(np.shape(pop['actions'][:,::-1,0]))
    for i in range(dimensions):
        fuel_sum += np.cumsum(np.abs(pop['actions'][:,::-1,i]), axis = 1)[:,::-1]
    nonzero_length = np.argmin(fuel_sum, axis = 1)
    
    temp_fuel_score = (result_weights[-1] - result_weights[-1] * (nonzero_length) / (np.size(pop['actions'],1)))
    temp_fuel_score[temp_fuel_score > (1-ideal_time_perc)*result_weights[-1]] = (1-ideal_time_perc)*result_weights[-1]
    pop['score'] += temp_fuel_score

    pop['score'] /= np.sum(result_weights)
    
    return pop

def pop_selection(pop, generation_nums, dimensions, steps):
    
    sorted_pop = np.argsort(pop['score'])[::-1]

    if generation_nums[0] == 0:
        elite_pop = np.array([])
    else:
        elite_pop = sorted_pop[:generation_nums[0]]
    if generation_nums[2] == 0:
        selected_pop = np.array([])
    else:
        selected_pop = sorted_pop[:generation_nums[2]]
    if generation_nums[1] == 0:       
        lucky_pop = np.array([])
    else:
        lucky_pop = np.random.choice(sorted_pop[generation_nums[2]:],size = generation_nums[1], replace = False)

    if generation_nums[3] == 0:
        mutated_pop = np.array([])
    else:
        mutated_pop = np.random.choice(sorted_pop[:],size = generation_nums[3], replace = False)

    actions = pop['actions'][np.concatenate((elite_pop,lucky_pop)).astype('int64')]

    pop = generate_children(pop, actions, generation_nums, selected_pop, mutated_pop, dimensions, steps)
    
    return pop, np.array([np.array(pop['score'][sorted_pop[0]]),pop['results'][sorted_pop[0]],sorted_pop[0]])
    
def generate_children(pop, actions, generation_nums, selected_pop, mutated_pop, dimensions, steps):
#    steps = 4
    
    mutated_actions = pop['actions'][mutated_pop,:,:]
    
#    mut_num = rdm.randint(1,generation_nums[4]+1)
#    indices = np.random.randint(0,np.size(mutated_actions[0,:,0]), size = (np.size(mutated_actions[:,0,0]),mut_num))
#
#    for i, mut_locs in enumerate(indices):
##        mutated_actions[i,mut_locs] = np.random.uniform(-1, 1, size = (mut_num, dimensions))
#        mutated_actions[i,mut_locs] = np.random.randint(-steps, steps+1, size = (mut_num, dimensions))/steps

    threshold_values = np.random.rand(np.shape(mutated_actions)[0],np.shape(mutated_actions)[1],np.shape(mutated_actions)[2])
    indices = np.where(threshold_values <= generation_nums[4]/100)
    
    for i in range(np.size(indices[0],0)):
        mutated_actions[indices[0][i],indices[1][i],indices[2][i]] = np.random.randint(-steps, steps+1, size = 1)/steps

        

#        if dimensions == 2:
#            mutated_actions[i,mut_locs,1] = np.random.uniform(-1, 1, size = mut_num)

#    random_selection_children_0 = np.random.randint(0,2,size = (int(len(selected_pop)/2),np.size(pop['actions'], axis = 1),dimensions))
#    if dimensions == 2:
#        random_selection_children_0[:,:,1] = random_selection_children_0[:,:,0]
    random_selection_children_0 = np.random.randint(1,np.size(pop['actions'], axis = 1)-1, size = generation_nums[2])
    selected_pop_0 = np.random.choice(selected_pop, size = int(len(selected_pop)/2), replace = False).astype('int64')
    selected_pop_1 = np.setdiff1d(selected_pop, selected_pop_0).astype('int64')
    
    children_actions_0 = pop['actions'][selected_pop,:,:]
    
    for i in range(np.size(selected_pop_0,0)):
        children_actions_0[i,:,:] = np.concatenate((pop['actions'][selected_pop_0[i],:random_selection_children_0[i],:],pop['actions'][selected_pop_1[i],random_selection_children_0[i]:,:]),axis = 0)
        children_actions_0[i+np.size(selected_pop_0,0),:,:] = np.concatenate((pop['actions'][selected_pop_1[i],:random_selection_children_0[i],:],pop['actions'][selected_pop_0[i],random_selection_children_0[i]:,:]),axis = 0)
   
#    children_actions_0 = pop['actions'][selected_pop_0,:,:]*random_selection_children_0 + pop['actions'][selected_pop_1,:,:]*random_selection_children_1
#    children_actions_1 = pop['actions'][selected_pop_1,:,:]*random_selection_children_0 + pop['actions'][selected_pop_0,:,:]*random_selection_children_1
    
    pop['actions'] = np.concatenate((actions,children_actions_0,mutated_actions), axis = 0)
    
    
    return pop

def run_GA_result(inputs,dimensions, goal, GA_parameters, parameters, generation_weights, samples):
    
    inputs = inputs/np.sum(inputs)
    
    pop, gen_count_stats = run_GA(dimensions, goal, GA_parameters, parameters, inputs, generation_weights, samples, steps)
    
    return gen_count_stats[0]

def run_GA_generation(inputs,dimensions, goal, GA_parameters, parameters, result_weights, samples):
    
    inputs[:-1] = inputs[:-1]/np.sum(inputs[:-1])
    
    pop, gen_count_stats = run_GA(dimensions, goal, GA_parameters, parameters, result_weights, inputs, samples, steps)
    
    return gen_count_stats[0]

def run_GA(dimensions, goal, GA_parameters, parameters, result_weights, generation_weights, samples, steps):
    
    [t_steps, pop_size, max_gen] = GA_parameters
    [vel_max, acc, dt, ideal_time_perc] = parameters
    
    print(generation_weights)
    
    generation_nums = get_generation_nums(t_steps, pop_size, generation_weights)
#    if generation_nums[4] == 0:
#        generation_nums[4] = 1
    
    theory_max = (np.sum(result_weights[:-1]) + result_weights[-1]*(1-ideal_time_perc))/(np.sum(result_weights))
    thresh_perf = 0.0
    
    COUNT = 0
    gen_count = np.zeros((2,samples))
    
    for KK in range(samples):  # Count for samples
        start_time = time.time()
        pop = initialise_population(dimensions, t_steps, pop_size, steps)
        
        for I in range(max_gen):
            pop = population_results(dimensions, pop, dt, acc, t_steps)
            pop = population_score(dimensions, pop, result_weights, goal, parameters)
            pop, best_performance = pop_selection(pop, generation_nums, dimensions, steps)

            if np.mod(I,max_gen/max_gen) == 0 or I == 0:
                print('t: %d, Pop: %d, Run: %1d, Max Perf: %3.3f' % (t_steps, pop_size, KK, theory_max))

                plt.figure(0)
                plt.clf()
                CS = plt.contourf(pop['actions'][:,:,0],levels=np.linspace(-1,1,steps*2 + 2))
                plt.colorbar(CS)
                plt.pause(0.1)

                if dimensions == 1:
                    print('Gen %1d Performance %3.3f, Distance = %3.1f, Velocity = %3.3f' % (I, best_performance[0], best_performance[1][0], best_performance[1][1]))
                elif dimensions == 2:
                    print('Gen %1d Performance %3.3f, Distance_x = %3.1f, Velocity_x = %3.1f, Distance_y = %3.3f, Velocity_y = %3.1f' % (I, best_performance[0], best_performance[1][0], best_performance[1][1], best_performance[1][2], best_performance[1][3]))

            if I == max_gen - 1:
                print('t: %d, Pop: %d, Run: %1d, Max Perf: %3.3f' % (t_steps, pop_size, KK, theory_max))
                if dimensions == 1:
                    print('Gen %1d Performance %3.3f, Distance = %3.1f, Velocity = %3.3f' % (I, best_performance[0], best_performance[1][0], best_performance[1][1]))
                elif dimensions == 2:
                    print('Gen %1d Performance %3.3f, Distance_x = %3.1f, Velocity_x = %3.1f, Distance_y = %3.3f, Velocity_y = %3.1f' % (I, best_performance[0], best_performance[1][0], best_performance[1][1], best_performance[1][2], best_performance[1][3]))
                
            if best_performance[0] >= theory_max-thresh_perf*theory_max:
                COUNT += 1
                if COUNT >= 2:
                    
                    sorted_pop = np.argsort(pop['score'])[::-1]
                    
#                    pop['actions'] = pop['actions'][sorted_pop]
                    pop['results'] = pop['results'][sorted_pop]
                    pop['score'] = pop['score'][sorted_pop] 
                    
                    if dimensions == 1:
                        print('Gen %1d Performance %3.3f, Distance = %3.1f, Velocity = %3.3f' % (I, best_performance[0], best_performance[1][0], best_performance[1][1]))
                    elif dimensions == 2:
                        print('Gen %1d Performance %3.3f, Distance_x = %3.1f, Velocity_x = %3.1f, Distance_y = %3.3f, Velocity_y = %3.1f' % (I, best_performance[0], best_performance[1][0], best_performance[1][1], best_performance[1][2], best_performance[1][3]))
                     
                    break
                
        elapsed_time = time.time() - start_time
#        print(elapsed_time)
        gen_count[0,KK] = I
        gen_count[1,KK] = elapsed_time
    gen_count_stats = np.zeros((3))
    gen_count_stats[:2] = np.median(gen_count,1)
    gen_count_stats[2] = np.sqrt(np.var(gen_count[0,:]))
    
    print(gen_count_stats[0])
    
    return pop, gen_count_stats

def set_goal(dimensions):
    
    if dimensions == 1:
        x_final = 100
        y_final = 0
        u_final = 0
        v_final = 0
    elif dimensions == 2:
        x_final = 100
        y_final = 70
        u_final = 0
        v_final = 0
    
    return np.array([x_final, u_final, y_final, v_final])

def set_parameters(dimensions, goal):
    
    dt = 1.0
    ideal_time_perc = 0.8
    t_steps = 20
    pop_size = 200
    max_gen = 10000
    
    vel_max, acc = get_acceleration(dimensions, goal, dt, t_steps, ideal_time_perc)
    
    x_weight = 0.2
    u_weight = 0.3
    t_weight = 0.75
    
    elite_weight = 0.45
    lucky_weight = 0.05
    mutation_weight = 0.2
    non_children_chance = (elite_weight + lucky_weight + mutation_weight)
    children_weight = np.ceil(non_children_chance) - non_children_chance
    mutation_chance = 0.2
    
    GA_parameters = np.array([int(t_steps), int(pop_size), int(max_gen)])
    parameters = np.array([vel_max, acc, dt, ideal_time_perc])
    
    result_weights = (np.array([x_weight,u_weight,t_weight])
            /(x_weight+u_weight+t_weight))
    
    generation_weights = (np.concatenate(((np.array([elite_weight,lucky_weight,children_weight,mutation_weight])
            /(elite_weight+lucky_weight+mutation_weight+children_weight)),np.array([mutation_chance])),axis=0))

    return GA_parameters, parameters, result_weights, generation_weights



if __name__ == "__main__":
    
    dimensions = 1
    samples = 1
    steps = 1
    
    goal = set_goal(dimensions)
        
    GA_parameters, parameters, result_weights, generation_weights = set_parameters(dimensions,goal)
    
#    result_weights = np.array([1.0, 1.0, 1.0])
#    result_weights = result_weights/np.sum(result_weights)
#    generation_weights = np.array([0.65725321, 0.1, 0.49855212, 0.20398279, 0.1])
#    generation_weights = generation_weights/np.sum(generation_weights)
    
    optimize_result_weights = False
    optimize_generation_weights = False
    
    
    if optimize_result_weights == True:
        orig_setting = result_weights
        xmin = np.ones((np.size(result_weights)))*0 + 0.001
        xmax = np.ones((np.size(result_weights)))*1.0 - 0.001
        
        bounds = [(low, high) for low, high in zip(xmin, xmax)]
        
        minimizer_kwargs = dict(method="L-BFGS-B", bounds = bounds, args = (dimensions, goal, 
                    GA_parameters, parameters, generation_weights, samples), tol=1e-2, options = {'maxiter':1,'disp':True})

        values = optim.basinhopping(run_GA_result, result_weights, minimizer_kwargs = minimizer_kwargs)
        new_result_weights = values.x
        print('Old:',orig_setting)
        print('New:',new_result_weights)
        
    elif optimize_generation_weights == True:
        orig_setting = generation_weights
        xmin = np.ones((np.size(generation_weights)))*0 + 0.001
        xmax = np.ones((np.size(generation_weights)))*1.0 - 0.001
        
        bounds = [(low, high) for low, high in zip(xmin, xmax)]
        
        minimizer_kwargs = dict(method="L-BFGS-B", bounds = bounds, args = (dimensions, goal, 
                    GA_parameters, parameters, result_weights, samples), tol=1e-2, options = {'maxiter':1,'disp':True})

        values = optim.basinhopping(run_GA_generation, generation_weights, minimizer_kwargs = minimizer_kwargs)

        new_generation_weights = values.x
        print('Old:',orig_setting)
        print('New:',new_generation_weights)
        
    else:
        pop, gen_count_stats  = run_GA(dimensions, goal, GA_parameters, parameters, result_weights, generation_weights, samples, steps)
    
    