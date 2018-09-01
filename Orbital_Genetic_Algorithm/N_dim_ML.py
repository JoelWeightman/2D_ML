# -*- coding: utf-8 -*-
"""
N-Dimensional Machine Learning Genetic Algorithm
Joel Weightman

A particle is accelerated at time t_0 at a value of acc_0, at t_1, the acceleration 
changes to acc_1. Cumulative sum is used for integration. No smoothing.

"""


import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.integrate as integ
from multiprocessing import Pool

def calculate_result(pop, acc, t_steps, dt, best_index, final, dimensions):
          
    if dimensions == 1:
        if final == False:
                    
            v, d = velocity_distance_1d(pop,dt,acc,t_steps)
                
            pop['results'] = np.concatenate((d,v), axis = 1)
        
            return pop
                
        else:
            
            v1 = np.zeros((np.shape(pop['actions'])[0],np.shape(pop['actions'])[1],1))
            
            v1[:,1:t_steps,0] = np.cumsum(pop['actions'][:,:-1]*dt*acc, axis = 1)[:,:,0]
            d1 = np.cumsum(v1*dt, axis = 1)
                
            plt.figure()
            plt.scatter(np.arange(np.size(pop['actions'][0,:])),d1[best_index,:,0], s = 1, c='red')
            plt.figure()
            plt.scatter(np.arange(np.size(pop['actions'][0,:])),v1[best_index,:,0], s = 1,c='black')
            
            return
        
    elif dimensions == 2:
        if final == False:
                        
            v, d_x, d_y = velocity_distance_2d(pop,dt,acc,t_steps)
                
            pop['results'] = np.concatenate((d_x, d_y, v), axis = 1)
        
            return pop
                
        else:
            
            v_x = np.zeros((np.shape(pop['actions'])[0],np.shape(pop['actions'])[1],1))
            v_y = np.zeros((np.shape(pop['actions'])[0],np.shape(pop['actions'])[1],1))
            
            v_x[:,1:t_steps,0] = np.cumsum(np.multiply(pop['actions'][:,:-1,0],acc*dt*np.cos(pop['actions'][:,:-1,1])), axis = 1)
            v_y[:,1:t_steps,0] = np.cumsum(np.multiply(pop['actions'][:,:-1,0],acc*dt*np.sin(pop['actions'][:,:-1,1])), axis = 1)
            
            d_x = integ.cumtrapz(v_x, dx = dt, axis = 1, initial = 0)
            d_y = integ.cumtrapz(v_y, dx = dt, axis = 1, initial = 0)
                
            
#            plt.figure(1)
#            plt.clf()
#            plt.scatter(np.arange(np.size(pop['actions'][0,:,0])),d_x[best_index,:,0], s = 1, c='red')
#            plt.scatter(np.arange(np.size(pop['actions'][0,:,0])),d_y[best_index,:,0], s = 1, c='blue')
#            plt.xlabel('Time')
#            plt.ylabel('Distance')
#            plt.pause(0.001)
#            plt.draw()
#            plt.figure(2)
#            plt.clf()
#            plt.scatter(np.arange(np.size(pop['actions'][0,:,0])),np.sqrt(v_x[best_index,:,0]**2 + v_y[best_index,:,0]**2), s = 1,c='black')
#            plt.xlabel('Time')
#            plt.ylabel('Velocity')
#            plt.pause(0.001)
#            plt.draw()
#            plt.figure(3)
#            plt.clf()
#            plt.scatter(d_x[best_index,:,0],d_y[best_index,:,0], s = 1,c='black')
#            plt.xlabel('Distance_X')
#            plt.ylabel('Distance_Y')
#            plt.pause(0.001)
#            plt.draw()
#            plt.figure(4)
#            plt.clf()
#            plt.scatter(np.arange(np.size(pop['actions'][0,:,0])),pop['actions'][best_index,:,1]*180/np.pi, s = 1,c='black')
#            plt.xlabel('Time')
#            plt.ylabel('Angle')
#            plt.pause(0.001)
#            plt.draw()
            
            plt.figure(1)
            plt.clf()
            plt.subplot(3,1,1)
            plt.scatter(np.arange(np.size(pop['actions'][0,:,0])),np.sqrt(v_x[best_index,:,0]**2 + v_y[best_index,:,0]**2), s = 1,c='black')
            plt.xlabel('Time')
            plt.ylabel('Velocity')
            plt.subplot(3,1,2)
            plt.scatter(d_x[best_index,:,0],d_y[best_index,:,0], s = 1,c='black')
            plt.xlabel('Distance_X')
            plt.ylabel('Distance_Y')
            plt.subplot(3,1,3)
            plt.scatter(np.arange(np.size(pop['actions'][0,:,0])),pop['actions'][best_index,:,1]*180/np.pi, s = 1,c='black')
            plt.xlabel('Time')
            plt.ylabel('Angle')
            plt.pause(0.001)
            plt.draw()
        
        return


def fitness(pop, weights, goals, thresh, v_max, dimensions):

    pop['actions'][:,-1,:] = 0.0
    loc_val = np.zeros((np.shape(pop['results'])[0]))
    goal_loc_val =np.zeros((np.shape(pop['results'])[0]))
    for dim in range(dimensions):
        
        loc_val += (pop['results'][:,dim] - goals[dim])**2
        goal_loc_val += goals[dim]**2
        
        
    loc_val = np.sqrt(loc_val) / np.sqrt(goal_loc_val)
    vel_val = np.abs(pop['results'][:,dimensions] - goals[dimensions]) / (v_max)
       
    goal_val = np.concatenate((loc_val[:,np.newaxis],vel_val[:,np.newaxis]), axis = 1)
    pop['score'] = (weights[0] - weights[0]*goal_val[:,0]) + (weights[1] - weights[1]*goal_val[:,1])
    
    fuel_sum = np.cumsum(np.abs(pop['actions'][:,::-1,0]), axis = 1)[:,::-1]
    nonzero_length = np.argmin(fuel_sum, axis = 1)
    
    temp_fuel_score = (weights[2] - weights[2] * (nonzero_length) / (np.size(pop['actions'],1)))# + (weights[3] - weights[3] * fuel_sum[:,0] / np.size(pop['actions'],1))
    temp_fuel_score[temp_fuel_score > 0.2*weights[2]] = 0.2*weights[2]
    pop['score'] += temp_fuel_score

#    print(nonzero_length.min())
##    print((np.size(pop['actions'],1)))
##    print(loc_val.min(), vel_val.min())
#    print((weights[0] - weights[0]*goal_val[0,0]), (weights[1] - weights[1]*goal_val[0,1]), (weights[2] - weights[2] * (nonzero_length[0]) / (np.size(pop['actions'],1))))
#    print((goal_val[0,0]), (goal_val[0,1]), (nonzero_length[0]) / (np.size(pop['actions'],1)))
#    print(weights[0], weights[1], weights[2]*0.2)
##    print((weights[0] - weights[0]*goal_val[:,0]).argmax(), (weights[1] - weights[1]*goal_val[:,1]).argmax(), (weights[2] - weights[2] * (nonzero_length) / (np.size(pop['actions'],1))).argmax())
##    print((weights[2] - weights[2] * (nonzero_length) / (np.size(pop['actions'],1))).max(), (weights[3] - weights[3] * fuel_sum[:,0] / np.size(pop['actions'],1)).max())
#    print(np.sum(weights))
    pop['score'] *= 1/np.sum(weights)
#    print(pop['score'].max())
    
    return pop

def init_population(pop, dimensions):
    
    if dimensions == 1:
        pop['actions'][:,:,0] = np.random.randint(-1, 2, size = np.shape(pop['actions'][:,:,0]))
    elif dimensions == 2:
        pop['actions'][:,:,0] = np.random.randint(0, 2, size = np.shape(pop['actions'][:,:,0]))
        pop['actions'][:,:,1] = np.random.uniform(0, 2*np.pi, size = np.shape(pop['actions'][:,:,1]))
#       pop['actions'][:,:,1] = np.random.randint(0, 6, size = np.shape(pop['actions'][:,:,1]))*60*(np.pi/180)
    
    return pop

def velocity_distance_1d(pop,dt,acc,t_steps):
    
    v = np.zeros((np.shape(pop['actions'])[0],np.shape(pop['actions'])[1],1))

    v[:,1:t_steps,0] = np.cumsum(np.multiply(pop['actions'][:,:-1],acc*dt), axis = 1)[:,:,0]
    d = np.trapz(v, dx = dt, axis = 1)
    
    return v[:,-1], d
    

def velocity_distance_2d(pop,dt,acc,t_steps):
    
    v_x = np.zeros((np.shape(pop['actions'])[0],np.shape(pop['actions'])[1],1))
    v_y = np.zeros((np.shape(pop['actions'])[0],np.shape(pop['actions'])[1],1))

    v_x[:,1:t_steps,0] = np.cumsum(np.multiply(pop['actions'][:,:-1,0],acc*dt*np.cos(pop['actions'][:,:-1,1])), axis = 1)
    v_y[:,1:t_steps,0] = np.cumsum(np.multiply(pop['actions'][:,:-1,0],acc*dt*np.sin(pop['actions'][:,:-1,1])), axis = 1)

    d_x = np.trapz(v_x, dx = dt, axis = 1)
    d_y = np.trapz(v_y, dx = dt, axis = 1)
    
    return np.sqrt(v_x[:,-1]**2 + v_y[:,-1]**2), d_x, d_y

def pop_selection(pop, selection_num, dimensions, angle_mutation):
    
    sorted_pop = np.argsort(pop['score'])[::-1]
    
    elite_pop = sorted_pop[:selection_num[0]]
    selected_pop = sorted_pop[:selection_num[1]]
    lucky_pop = np.random.choice(sorted_pop[selection_num[1]:],size = selection_num[2], replace = False)
    mutated_pop = np.random.choice(selected_pop, size = selection_num[3], replace = False)
#    print(selection_num[3])
    selected_pop = np.setdiff1d(selected_pop,mutated_pop)
    
    actions = pop['actions'][np.concatenate((elite_pop,lucky_pop))]
    
    pop = generate_children(pop, actions, selection_num, selected_pop, mutated_pop, dimensions, angle_mutation)
    
    return pop, [np.array(pop['score'][sorted_pop[0]]),pop['results'][sorted_pop[0]],sorted_pop[0]]

def generate_children(pop, actions, selection_num, selected_pop, mutated_pop, dimensions, angle_mutation):

    mutated_actions = pop['actions'][mutated_pop,:,:]
    mut_num = np.random.randint(1,selection_num[4]+1,size = 1)[0]
    indices = np.random.randint(0,np.size(mutated_actions[0,:,0]), size = (np.size(mutated_actions[:,0,0]),mut_num))
    for i, mut_locs in enumerate(indices):
        if dimensions == 1:
            mutated_actions[i,mut_locs,0] = np.random.randint(-1, 2, size = mut_num)
        elif dimensions == 2:
            mutated_actions[i,mut_locs,0] = np.random.randint(0, 2, size = mut_num)
    if dimensions == 2:
        mut_num = np.random.randint(1,selection_num[4]+1,size = 1)[0]
        indices = np.random.randint(0,np.size(mutated_actions[0,:,0]), size = (np.size(mutated_actions[:,0,0]),mut_num))
        for i, mut_locs in enumerate(indices):
#            mutated_actions[i,mut_locs,1] = np.random.uniform(0, 2*np.pi, size = mut_num)
            mutated_actions[i,mut_locs,1] = mutated_actions[i,mut_locs,1]*np.random.uniform(1-angle_mutation, 1+angle_mutation, size = mut_num)
#            mutated_actions[i,mut_locs,1] = np.random.randint(0, 6, size = mut_num)*60*(np.pi/180)
    
    random_selection_children_0 = np.random.randint(0,2,size = (int(len(selected_pop)/2),np.size(pop['actions'][0,:,0]),dimensions))
    random_selection_children_1 = 1 - random_selection_children_0
    
    selected_pop_0 = np.random.choice(selected_pop, size = int(len(selected_pop)/2), replace = False)
    selected_pop_1 = np.setdiff1d(selected_pop, selected_pop_0)
    
    children_actions_0 = pop['actions'][selected_pop_0,:,:]*random_selection_children_0 + pop['actions'][selected_pop_1,:,:]*random_selection_children_1
    children_actions_1 = pop['actions'][selected_pop_1,:,:]*random_selection_children_0 + pop['actions'][selected_pop_0,:,:]*random_selection_children_1
    
    pop['actions'] = np.concatenate((actions,mutated_actions,children_actions_0,children_actions_1), axis = 0)

    return pop

def run(A_l, A_v, A_f, A_t, perc_elite, perc_lucky, perc_mutation, mutation_chance, pop_size, generations, t_steps, loc_des_x, loc_des_y, vel_des, ideal_time_perc, dt, dimensions, samples, angle_mutation):
    
    plt.close("all")
    
    # Threshold Values
    thresh_loc = 0.0
    thresh_vel = 0.0
    thresh_time = 0.0
    thresh_perf = 1e-5
    
    thresh = np.array([thresh_loc, thresh_vel, thresh_time])
    A_f = 0.0
    weights = np.array([A_l, A_v, A_t, A_f])
    if dimensions == 1:
        goals = np.array([loc_des_x, vel_des, ideal_time_perc])
    elif dimensions == 2:
        goals = np.array([loc_des_x, loc_des_y, vel_des, ideal_time_perc])
    
    ## Conditions to set
    gen_count = np.zeros((2,samples))
    
    for KK in range(samples):
        start_time = time.time()
        
        pop = dict({'actions': np.zeros((pop_size,t_steps,dimensions)), 'results':np.zeros((pop_size,dimensions + 1)), 'score':np.zeros((pop_size))})
        
        ## Calculated Parameters
        pop_num_elite = int(perc_elite*pop_size)
        if pop_num_elite < 2:
            pop_num_elite = 2
        pop_num_mutation = int(perc_mutation*pop_size)
        if pop_num_mutation < 1:
            pop_num_mutation = 1
        pop_num_lucky = int((pop_size*perc_lucky))
        if pop_num_lucky < 1:
            pop_num_lucky = 1
        total_non_children = pop_num_lucky + pop_num_elite + pop_num_mutation
        pop_num_selected = pop_size - total_non_children + pop_num_mutation
        if np.mod(pop_num_selected - pop_num_mutation,2) != 0:
            pop_num_selected += 1
            pop_num_elite -= 1
            
        mutation_gene_num = int(mutation_chance*t_steps)
        selection_num = np.array([pop_num_elite, pop_num_selected, pop_num_lucky, pop_num_mutation, mutation_gene_num])
        if dimensions == 1:
            v_max = (loc_des_x)/(dt*t_steps*ideal_time_perc/2)
        elif dimensions == 2:
            v_max = np.sqrt(loc_des_x**2 + loc_des_y**2)/(dt*t_steps*ideal_time_perc/2)
            
        acc = v_max/(dt*t_steps*ideal_time_perc/2)
        theory_max = (A_l + A_v + A_t*(1-ideal_time_perc))/(A_l + A_v + A_t)
        
        COUNT = 0
        ## First iteration
        pop = init_population(pop, dimensions)
        ## Calculate Generations until convergence or theory max               
        for I in range(generations):
            
            pop = calculate_result(pop, acc, t_steps, dt, [], False, dimensions)
            pop = fitness(pop, weights, goals, thresh, v_max, dimensions)
            pop, best_performance = pop_selection(pop, selection_num, dimensions, angle_mutation)
            
            if np.mod(I,1000) == 0 and I != 0:
                print('t: %d, Pop: %d, Run: %1d, Max Perf: %3.3f' % (t_steps,pop_size,KK, theory_max))
                if dimensions == 1:
                    print('Gen %1d Performance %3.3f, Distance = %3.1f, Velocity = %3.3f' % (I, best_performance[0], best_performance[1][0], best_performance[1][1]))
                elif dimensions == 2:
                    print('Gen %1d Performance %3.3f, Distance_x = %3.1f, Distance_y = %3.1f, Velocity = %3.3f' % (I, best_performance[0], best_performance[1][0], best_performance[1][1], best_performance[1][2]))
#                calculate_result(pop, acc, t_steps, dt, 0, True, dimensions)
            if I == generations - 1:
                print('t: %d, Pop: %d, Run: %1d, Max Perf: %3.3f' % (t_steps,pop_size,KK, theory_max))
                if dimensions == 1:
                    print('Gen %1d Performance %3.3f, Distance = %3.1f, Velocity = %3.3f' % (I, best_performance[0], best_performance[1][0], best_performance[1][1]))
                elif dimensions == 2:
                    print('Gen %1d Performance %3.3f, Distance_x = %3.1f, Distance_y = %3.1f, Velocity = %3.3f' % (I, best_performance[0], best_performance[1][0], best_performance[1][1], best_performance[1][2]))
#                calculate_result(pop, acc, t_steps, dt, 0, True, dimensions)
                
            if best_performance[0] >= theory_max-thresh_perf:
                COUNT += 1
                if COUNT >= 2:
                    if dimensions == 1:
                        print('Gen %1d Performance %3.3f, Distance = %3.1f, Velocity = %3.3f' % (I, best_performance[0], best_performance[1][0], best_performance[1][1]))
                    elif dimensions == 2:
                        print('Gen %1d Performance %3.3f, Distance_x = %3.1f, Distance_y = %3.1f, Velocity = %3.3f' % (I, best_performance[0], best_performance[1][0], best_performance[1][1], best_performance[1][2]))
                    break
        
#        print(A_l, A_v, A_t)
        elapsed_time = time.time() - start_time
        gen_count[0,KK] = I
        gen_count[1,KK] = elapsed_time

    gen_count_stats = np.zeros((3))
    gen_count_stats[:2] = np.mean(gen_count,1)
    gen_count_stats[2] = np.sqrt(np.var(gen_count[1,:]))
    
    return gen_count_stats, pop, best_performance, acc, dt

def n_d_runfile(dimensions = 1, loc_des_x = 1, loc_des_y = 1, vel_des = 1, t_steps = 1, pop_size = 1, A_l = 1, A_v = 1, A_f = 1, A_t = 1, perc_elite = 1, perc_lucky = 1, perc_mutation = 1, perc_selected = 1, mutation_chance = 1, angle_mutation = 1, samples = 1, generations = 1):

#    print('HELLO')
    dimensions = 1
    dt = 1.0
    ideal_time_perc = 0.8
    perc_elite, perc_lucky, perc_mutation, perc_selected = np.array([perc_elite, perc_lucky, perc_mutation, perc_selected])/(perc_elite + perc_lucky + perc_mutation + perc_selected)

    gen_count_stats, pop, best_performance, acc, dt = run(A_l, A_v, A_f, A_t, perc_elite, perc_lucky, perc_mutation, mutation_chance, pop_size, generations, t_steps, loc_des_x, loc_des_y, vel_des, ideal_time_perc, dt, dimensions, samples, angle_mutation)
    
    calculate_result(pop, acc, t_steps, dt, best_performance[2], True, dimensions)

    return gen_count_stats[0]


def fitness_ML(pop_ML):
    
    pop_ML['score'] = np.exp(-(pop_ML['results']/1000))
    
    return pop_ML

def init_population_ML(pop_ML):
    
    pop_ML['actions'] = np.random.uniform(0.10,0.90,size = (np.shape(pop_ML['actions'])[0],np.shape(pop_ML['actions'])[1]))
    
    return pop_ML

def calculate_ML(pop_ML, current_gen, ML_settings):
    
    start_time = time.time()
    gen_stats = np.zeros((np.size(pop_ML['actions'],0)))
    pool = Pool(10)
    
    ML_dimensions, ML_loc_des_x, ML_loc_des_y, ML_vel_des, ML_t_steps, ML_pop_size_max, ML_sample, ML_generations = ML_settings
  
    A = []
       
    for i in range(np.size(pop_ML['actions'],0)):
        A.append((ML_dimensions,ML_loc_des_x,ML_loc_des_y,ML_vel_des,ML_t_steps,int(pop_ML['actions'][i,0]*ML_pop_size_max),pop_ML['actions'][i,1],pop_ML['actions'][i,2],pop_ML['actions'][i,3],pop_ML['actions'][i,4],pop_ML['actions'][i,5],pop_ML['actions'][i,6],pop_ML['actions'][i,7],pop_ML['actions'][i,8],pop_ML['actions'][i,9],pop_ML['actions'][i,10],ML_sample,ML_generations))
#    print(A)
    print('Gen = %d' %(current_gen))
    gen_stats = np.zeros((np.shape(A)[0]))
#    for i in range(np.shape(A)[0]):
#        dimensions, loc_des_x, loc_des_y, vel_des, t_steps, pop_size, A_l, A_v, A_f, A_t, perc_elite, perc_lucky, perc_mutation, perc_selected, mutation_chance, angle_mutation, samples, generations = A[i]
#        gen_stats[i] = n_d_runfile(dimensions, loc_des_x, loc_des_y, vel_des, t_steps, pop_size, A_l, A_v, A_f, A_t, perc_elite, perc_lucky, perc_mutation, perc_selected, mutation_chance, angle_mutation, samples, generations)
#    print(resultant_gen)
    for i, result in enumerate(pool.starmap(n_d_runfile, A)):
        print(current_gen, i, result)
        gen_stats[i] = result
        
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    
    pop_ML['results'] = gen_stats
    pool.close()
    print(np.mean(gen_stats))
    
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

def run_ML(generations, ML_settings, already_started = 0, pop_ML = None):

    pop_size = 50
    t_steps = 11
    
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
    
    time_to_locvel_ratio = 0.49
    dist_to_vel_ration = 0.5
    
    if already_started == 0:
        print('Starting')
        pop_ML = dict({'actions': np.zeros((pop_size,t_steps)), 'results':np.zeros((pop_size)), 'score':np.zeros((pop_size))})
        pop_ML = init_population_ML(pop_ML)      
    else:
        print('Already Started, but Continuing')
    
    for I in range(generations):
        
#        pop_ML['actions'][:,1] = dist_to_vel_ration*pop_ML['actions'][:,2]
#        
        vel_loc_weight = (pop_ML['actions'][:,1] + pop_ML['actions'][:,2])
        bad_time_weights = pop_ML['actions'][:,4] > vel_loc_weight*time_to_locvel_ratio
        pop_ML['actions'][bad_time_weights,4] = (pop_ML['actions'][bad_time_weights,1] + pop_ML['actions'][bad_time_weights,2])*time_to_locvel_ratio

        pop_ML = calculate_ML(pop_ML, I, ML_settings)
        pop_ML = fitness_ML(pop_ML)
        pop_ML, best_performance = pop_selection_ML(pop_ML, selection_num_ML)
        
        print('BEST PERFORMANCE AT GEN %d: %3.3f' %(I, best_performance[0]))
        
        filename = 'Population_ML_Gen_Temp.npy'
        np.save(filename, pop_ML)
       
    return pop_ML, best_performance

def start_ML_ML_optimization(generations):
           
    ML_dimensions = 1
    ML_loc_des_x = 100
    ML_loc_des_y = 100*np.tan(60*np.pi/180)
    ML_vel_des = 0
    ML_t_steps = 50
    ML_pop_size_max = 1000
    ML_sample = 10
    ML_generations = 50000
    
    ML_settings = [ML_dimensions, ML_loc_des_x, ML_loc_des_y, ML_vel_des, ML_t_steps, ML_pop_size_max, ML_sample, ML_generations]
    
    already_started = 0
    
    if already_started == 0:
        pop_ML, best_performance = run_ML(generations, ML_settings, already_started, [])
    else:
        filename = 'ML_t_' + str(ML_t_steps) + '_samp_' + str(ML_sample) + '_pop_' + str(ML_pop_size_max) + '_settings_1.npy'
        pop_ML = np.load(filename).item()
        filename = 'ML_t_' + str(ML_t_steps) + '_samp_' + str(ML_sample) + '_pop_' + str(ML_pop_size_max) + '_settings_2.npy'
        ML_settings = np.load(filename).item()
        pop_ML, best_performance = run_ML(generations, ML_settings, already_started, pop_ML)
    
    filename = 'ML_t_' + str(ML_t_steps) + '_samp_' + str(ML_sample) + '_pop_' + str(ML_pop_size_max) + '_settings_1.npy'
    np.save(filename, pop_ML)
    filename = 'ML_t_' + str(ML_t_steps) + '_samp_' + str(ML_sample) + '_pop_' + str(ML_pop_size_max) + '_settings_2.npy'
    np.save(filename, ML_settings)

    return

if __name__ == "__main__":
       
    plt.close('all')
    ### Design Values
    
    optimize_ML = 1
    generations = 100
    
    if optimize_ML == 0:
    
        dimensions = 1
        loc_des_x = 100
        loc_des_y = 100*np.tan(60*np.pi/180)
        vel_des = 0
        
        t_steps = 20
        ideal_time_perc = 0.8
        dt = 1.0
        
        pop_size = 100
        A_l = 0.3
        A_v = 0.6
        A_f = 0
        A_t = 0.2
        
        ## Other Set Params
        perc_elite = 0.2
        perc_lucky = 0.15
        perc_mutation = 0.2
        perc_selected = 0.45
        mutation_chance = 0.4
        angle_mutation = 0.1
        
        generations = 10000
        samples = 1
        
        mean_generations = n_d_runfile(dimensions, loc_des_x, loc_des_y, vel_des, t_steps, pop_size, A_l, A_v, A_f, A_t, perc_elite, perc_lucky, perc_mutation, perc_selected, mutation_chance, angle_mutation, samples, generations)
    
    elif optimize_ML == 1:
        start_ML_ML_optimization(generations)