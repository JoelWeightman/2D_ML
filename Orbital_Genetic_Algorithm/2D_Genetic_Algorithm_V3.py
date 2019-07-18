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
import scipy.integrate as integ

def plot_actions(pop, acc, t_steps, dt, dimensions):
    
    v_x = np.zeros((np.shape(pop['actions'])[0],np.shape(pop['actions'])[1],1))
    v_y = np.zeros((np.shape(pop['actions'])[0],np.shape(pop['actions'])[1],1))
    
    v_x[:,1:t_steps,0] = np.cumsum(np.multiply(pop['actions'][:,:-1,0],acc*dt*np.cos(pop['actions'][:,:-1,1])), axis = 1)
    v_y[:,1:t_steps,0] = np.cumsum(np.multiply(pop['actions'][:,:-1,0],acc*dt*np.sin(pop['actions'][:,:-1,1])), axis = 1)
    
    d_x = integ.cumtrapz(v_x, dx = dt, axis = 1, initial = 0)
    d_y = integ.cumtrapz(v_y, dx = dt, axis = 1, initial = 0)
    
    best_index = 0
    
    plt.figure(2)
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
    plt.scatter(np.arange(np.size(pop['actions'][0,:,0])),pop['actions'][best_index,:,0]*360, s = 1,c='red')
    plt.xlabel('Time')
    plt.ylabel('Angle')
    plt.ylim(0,360)
    plt.pause(0.001)
    plt.draw()
        
    return

def get_acceleration(dimensions, goal, dt, t_steps, ideal_time_perc):
 
    [x_final,u_final,y_final,v_final] = goal
    
    if dimensions == 1:
        u_max = (x_final)/(dt*t_steps*ideal_time_perc/2)+u_final
        u_acc = u_max/(dt*t_steps*ideal_time_perc/2)
        
    elif dimensions == 2:
        u_max = np.sqrt(x_final**2 + y_final**2)/(dt*t_steps*ideal_time_perc/2)+u_final+v_final
        u_acc = u_max/(dt*t_steps*ideal_time_perc/2)
    
    u_max = 10
    u_acc = u_max/(dt*t_steps*ideal_time_perc/2)
    
    print(u_max)
    return u_max, u_acc          

def initialise_population(dimensions, t_steps, pop_size, steps, deg_steps):
    
    pop = dict({'actions': np.zeros((pop_size,t_steps,dimensions)), 'results':np.zeros((pop_size,dimensions*2)), 'score':np.zeros((pop_size))})

    pop['actions'] = initial_values(dimensions,pop['actions'],steps, deg_steps)

    return pop

def initial_values(dimensions, actions, steps, deg_steps):
#    steps = 4
    if dimensions == 1:
#        actions[:,:,0] = np.random.uniform(-1, 1, size = np.shape(actions[:,:,0]))
        actions[:,:,0] = np.random.randint(-steps, steps+1, size = np.shape(actions[:,:,0]))/steps
    elif dimensions == 2:
#        actions[:,:,0] = np.random.uniform(-1, 1, size = np.shape(actions[:,:,0]))
        actions[:,:,0] = np.random.randint(0, steps+1, size = np.shape(actions[:,:,0]))/steps
#        actions[:,:,1] = np.random.uniform(-1, 1, size = np.shape(actions[:,:,0]))
#        actions[:,:,1] = np.random.randint(-steps, steps+1, size = np.shape(actions[:,:,0]))/steps
        actions[:,:,1] = np.random.randint(0, 360/deg_steps, size = np.shape(actions[:,:,1]))*deg_steps*(np.pi/180)

    
    return actions

def get_generation_nums(t_steps, pop_size, generation_weights):
    
    elite_num = 2#np.round(generation_weights[0]*pop_size).astype('int64')
    lucky_num = 2#np.round(generation_weights[1]*pop_size).astype('int64')
    
    generation_weights[2:4] = generation_weights[2:4]/np.sum(generation_weights[2:4])*(pop_size-elite_num-lucky_num)/pop_size
    
    children_num = (np.round(generation_weights[2]*pop_size/2)*2).astype('int64')
    mutation_num = np.round(generation_weights[3]*pop_size).astype('int64')
    
    if children_num < 2:
        mutation_num -= children_num
        children_num = 2
    
    mutated_gene_max = np.round(generation_weights[-1]*1000).astype('int64')
    
    if elite_num + lucky_num + children_num + mutation_num < pop_size:
        mutation_num += pop_size - (elite_num + lucky_num + children_num + mutation_num)
    elif elite_num + lucky_num + children_num + mutation_num > pop_size:
        mutation_num += pop_size - (elite_num + lucky_num + children_num + mutation_num)
    
    generation_nums = np.array([elite_num,lucky_num,children_num,mutation_num,mutated_gene_max]).astype('int64')
    print(generation_nums)
    return generation_nums

def population_results(dimensions, pop, dt, acc, t_steps):
    
    if dimensions == 1:
                    
        v, d = velocity_distance_1d(pop,dt,acc,t_steps)  
        pop['results'] = np.concatenate((d,v), axis = 1)
       
    elif dimensions == 2:
    
        u, v, d_x, d_y = velocity_distance_2d(pop,dt,acc,t_steps)
#        print(np.shape(u),np.shape(v),np.shape(d_x),np.shape(d_y))
            
        pop['results'] = np.concatenate((d_x, u, d_y, v), axis = 1)


    return pop

def velocity_distance_1d(pop, dt, acc, t_steps):
    
    v = np.zeros((np.shape(pop['actions'])[0],np.shape(pop['actions'])[1],1))

    v[:,1:t_steps,0] = np.cumsum(np.multiply(pop['actions'][:,:-1],acc*dt), axis = 1)[:,:,0]
    d = np.trapz(v, dx = dt, axis = 1)
    
    return v[:,-1], d
    

def velocity_distance_2d(pop, dt, acc, t_steps):
    
    v_x = np.zeros((np.shape(pop['actions'])[0],np.shape(pop['actions'])[1],1))
    v_y = np.zeros((np.shape(pop['actions'])[0],np.shape(pop['actions'])[1],1))
    
    v_x[:,1:t_steps,0] = np.cumsum(np.multiply(pop['actions'][:,:-1,0],acc*dt*np.cos(pop['actions'][:,:-1,1])), axis = 1)
    v_y[:,1:t_steps,0] = np.cumsum(np.multiply(pop['actions'][:,:-1,0],acc*dt*np.sin(pop['actions'][:,:-1,1])), axis = 1)
    
    d_x = integ.cumtrapz(v_x, dx = dt, axis = 1, initial = 0)
    d_y = integ.cumtrapz(v_y, dx = dt, axis = 1, initial = 0)
    
    return v_x[:,-1], v_y[:,-1], d_x[:,-1], d_y[:,-1]

def population_score(dimensions, pop, result_weights, goal, parameters):

    [vel_max, acc, dt, ideal_time_perc] = parameters
    
    pop['actions'][:,-1,:] = 0.0
    
    loc_val = np.zeros((np.shape(pop['results'])[0]))
    vel_val = np.zeros((np.shape(pop['results'])[0]))
    goal_loc_val = np.zeros((np.shape(pop['results'])[0]))

    for dim in np.arange(0,dimensions+1,2):
        
        loc_val += (pop['results'][:,dim] - goal[dim])**2
                
        vel_val += np.abs(pop['results'][:,dim+1] - goal[dim+1])
        
    goal_loc_val = 1
    goal_vel_val = 1
 
    loc_val = np.sqrt(loc_val) / goal_loc_val
    vel_val = (vel_val) / (goal_vel_val)

    pop['score'] = (result_weights[0] - result_weights[0]*loc_val) + (result_weights[1] - result_weights[1]*vel_val)
    
#    fuel_sum = np.zeros(np.shape(pop['actions'][:,::-1,0]))
#    for i in range(dimensions):
#        fuel_sum += np.cumsum(np.abs(pop['actions'][:,::-1,i]), axis = 1)[:,::-1]
#    nonzero_length = np.argmin(fuel_sum, axis = 1)
#    
#    temp_fuel_score = (result_weights[-1] - result_weights[-1] * (nonzero_length) / (np.size(pop['actions'],1)))
##    temp_fuel_score[temp_fuel_score > (1-ideal_time_perc)*result_weights[-1]] = (1-ideal_time_perc)*result_weights[-1]
#    pop['score'] += temp_fuel_score

    pop['score'] /= np.sum(result_weights)
    
    return pop

def pop_selection(pop, generation_nums, dimensions, steps, deg_steps):
    
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
        lucky_pop = np.random.choice(sorted_pop[:],size = generation_nums[1], replace = False)
        
    if generation_nums[3] == 0:
        mutated_pop = np.array([])
    else:
        mutated_pop = np.random.choice(sorted_pop[:],size = generation_nums[3], replace = False)

    actions = pop['actions'][np.concatenate((elite_pop,lucky_pop)).astype('int64')]

    pop = generate_children(pop, actions, generation_nums, selected_pop, mutated_pop, dimensions, steps, deg_steps)
    
    return pop, np.array([np.array(pop['score'][sorted_pop[0]]),pop['results'][sorted_pop[0]],sorted_pop[0]])
    
def generate_children(pop, actions, generation_nums, selected_pop, mutated_pop, dimensions, steps, deg_steps):

    
    ### Mutate Children
    if mutated_pop.size != 0:
        mutated_actions = pop['actions'][mutated_pop,:,:]
        
        threshold_values = np.random.rand(np.shape(mutated_actions)[0],np.shape(mutated_actions)[1],np.shape(mutated_actions)[2])
        indices = np.where(threshold_values <= generation_nums[-1]/1000)
        
        for i in range(np.size(indices[0],0)):
            if indices[2][i] == 1:
                mutated_actions[indices[0][i],indices[1][i],indices[2][i]] = np.random.randint(0, 360/deg_steps, size = 1)*deg_steps*(np.pi/180)
            else:
                mutated_actions[indices[0][i],indices[1][i],indices[2][i]] = np.random.randint(-0**(dimensions-1), steps+1, size = 1)/steps
    
    if selected_pop.size != 0:
        ## Cross Over Children
        random_selection_children_0 = np.random.randint(1,np.size(pop['actions'], axis = 1)-1, size = generation_nums[2])
        selected_pop_0 = np.random.choice(selected_pop, size = int(len(selected_pop)/2), replace = False).astype('int64')
        selected_pop_1 = np.setdiff1d(selected_pop, selected_pop_0).astype('int64')
        
        children_actions_0 = pop['actions'][selected_pop,:,:]
        
        for i in range(np.size(selected_pop_0,0)):
            children_actions_0[i,:,:] = np.concatenate((pop['actions'][selected_pop_0[i],:random_selection_children_0[i],:],pop['actions'][selected_pop_1[i],random_selection_children_0[i]:,:]),axis = 0)
            children_actions_0[i+np.size(selected_pop_0,0),:,:] = np.concatenate((pop['actions'][selected_pop_1[i],:random_selection_children_0[i],:],pop['actions'][selected_pop_0[i],random_selection_children_0[i]:,:]),axis = 0)
            
    if mutated_pop.size == 0:
        pop['actions'] = np.concatenate((actions,children_actions_0), axis = 0)
        
    elif selected_pop.size == 0:
        pop['actions'] = np.concatenate((actions,mutated_actions), axis = 0)
    else:
        pop['actions'] = np.concatenate((actions,children_actions_0,mutated_actions), axis = 0)
        
    return pop

def run_GA_result(inputs, dimensions, goal, GA_parameters, parameters, generation_weights, samples, steps, deg_steps):
    
    centre_values = np.array([0.5,0.5,0.5])
    for iteration in range(3):
        step_size = 5
        epsilon = 0.001
        
        list_1 = np.linspace(-0.5,0.5,step_size)/step_size**iteration+centre_values[0]
        list_1[list_1 <= epsilon] = epsilon
        list_1[list_1 >= (1-epsilon)] = (1-epsilon)
        list_2 = np.linspace(-0.5,0.5,step_size)/step_size**iteration+centre_values[1]
        list_2[list_2 <= epsilon] = epsilon
        list_2[list_2 >= (1-epsilon)] = (1-epsilon)
        list_3 = np.array([0.0])#np.linspace(-0.5,0.5,step_size)/step_size**iteration+centre_values[2]
#        list_3[list_3 <= epsilon] = epsilon
#        list_3[list_3 >= (1-epsilon)] = (1-epsilon)
          
        scored = np.zeros((step_size,step_size,1))
        
        for I,i in enumerate(list_1):
            for J,j in enumerate(list_2):
                for K,k in enumerate(list_3):
                    
                    inputs = np.array([i,j,k])
        
                    inputs = inputs/np.sum(inputs)
                    print(inputs)
                    pop, gen_count_stats = run_GA(dimensions, goal, GA_parameters, parameters, inputs, generation_weights, samples, steps, deg_steps, True)
#                    print(iteration,I,J,K,gen_count_stats[0],gen_count_stats[1])
#                    scored[I,J,K] = gen_count_stats[0] + 2*gen_count_stats[1]
                    print(iteration,I,J,K,pop['score'][0])
                    scored[I,J,K] = pop['score'][0]
      
        min_val_loc = np.where(scored == np.max(scored))
        centre_values = np.array([list_1[min_val_loc[0]],list_2[min_val_loc[1]],list_3[min_val_loc[2]]])
    
    return scored, centre_values

def run_GA_generation(inputs,dimensions, goal, GA_parameters, parameters, result_weights, samples, steps, deg_steps):
    
    centre_values = np.array([0.5,0.5,0.5,0.5,0.05])
    mutation_chance_max = 0.1
    for iteration in range(3):
        step_size = 5
        epsilon = 0.001
        
        list_1 = np.array([0.5])#np.linspace(-0.5,0.5,step_size)/step_size**iteration+centre_values[0]
        list_1[list_1 <= epsilon] = epsilon
        list_1[list_1 >= (1-epsilon)] = (1-epsilon)
        list_2 = np.array([0.5])# np.linspace(-0.5,0.5,step_size)/step_size**iteration+centre_values[1]
        list_2[list_2 <= epsilon] = epsilon
        list_2[list_2 >= (1-epsilon)] = (1-epsilon)
        list_3 = np.linspace(-0.5,0.5,step_size)/step_size**iteration+centre_values[2]
        list_3[list_3 <= epsilon] = epsilon
        list_3[list_3 >= (1-epsilon)] = (1-epsilon)
        list_4 = np.linspace(-0.5,0.5,step_size)/step_size**iteration+centre_values[3]
        list_4[list_4 <= epsilon] = epsilon
        list_4[list_4 >= (1-epsilon)] = (1-epsilon)
        list_5 = np.linspace(-0.5,0.5,step_size)/(10*step_size**iteration)+centre_values[4]
        list_5[list_5 <= epsilon] = epsilon
        list_5[list_5 >= (1-mutation_chance_max)] = (1-mutation_chance_max)
          
        scored = np.zeros((1,1,step_size,step_size,step_size))
        
        for I,i in enumerate(list_1):
            for J,j in enumerate(list_2):
                for K,k in enumerate(list_3):
                    for M,m in enumerate(list_4):
                        for N,n in enumerate(list_5):
    #                        k1 = 1 - i - j
    #                        if k1*GA_parameters[1] < 1:
    #                            k1 = 0
                            inputs = np.array([i,j,k,m,n])
                
#                            inputs[:-1] = inputs[:-1]/np.sum(inputs[:-1])
                            print(inputs)
                            pop, gen_count_stats = run_GA(dimensions, goal, GA_parameters, parameters, result_weights, inputs, samples, steps, deg_steps, True)
#                            print(iteration,I,J,K,M,N,gen_count_stats[0],gen_count_stats[1])
#                            scored[I,J,K,M,N] = gen_count_stats[0] + 2*gen_count_stats[1]
                            print(iteration,I,J,K,M,N,pop['score'][0])
                            scored[I,J,K,M,N] = pop['score'][0]
        
        min_val_loc = np.where(scored == np.max(scored))
        centre_values = np.array([list_1[min_val_loc[0]],list_2[min_val_loc[1]],list_3[min_val_loc[2]],list_4[min_val_loc[3]],list_5[min_val_loc[4]]])
    
    return scored, centre_values

def run_GA(dimensions, goal, GA_parameters, parameters, result_weights, generation_weights, samples, steps, deg_steps, running_ML = False):
    
    [t_steps, pop_size, max_gen] = GA_parameters
    [vel_max, acc, dt, ideal_time_perc] = parameters
    
    generation_nums = get_generation_nums(t_steps, pop_size, generation_weights)
    orig_mutation_chance = np.copy(generation_nums[-1])
#    if generation_nums[4] == 0:
#        generation_nums[4] = 1
    
    theory_max = (np.sum(result_weights[:-1]) + result_weights[-1]*(1-ideal_time_perc))/(np.sum(result_weights))
    thresh_perf = 1e-4
    
    stall_count = 0
    gen_count = np.zeros((2,samples))
    
    
    for KK in range(samples):  # Count for samples
        start_time = time.time()
        pop = initialise_population(dimensions, t_steps, pop_size, steps, deg_steps)
        old_best = np.copy(pop['actions'][0,:,0]*pop['actions'][0,:,1])
        
        for I in range(max_gen):
            pop = population_results(dimensions, pop, dt, acc, t_steps)
            pop = population_score(dimensions, pop, result_weights, goal, parameters)
            pop, best_performance = pop_selection(pop, generation_nums, dimensions, steps, deg_steps)

            if running_ML == False:
                if np.mod(I,max_gen/10) == 0 or I == 0:
                    print('t: %d, Pop: %d, Run: %1d, Max Perf: %3.3f' % (t_steps, pop_size, KK, theory_max))
    
                    if True:
                        plt.figure(0)
                        plt.clf()
                        CS = plt.imshow(pop['actions'][:,:,0]*pop['actions'][:,:,1], aspect='auto')
                        plt.colorbar(CS)
                        plt.pause(0.01)
                        
                    if True:
                        plot_actions(pop, acc, t_steps, dt, dimensions)
                        
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
                
#            if best_performance[0] >= theory_max-thresh_perf*theory_max:
#                    
#                    sorted_pop = np.argsort(pop['score'])[::-1]
#                    
#                    pop['results'] = pop['results'][sorted_pop]
#                    pop['score'] = pop['score'][sorted_pop] 
#                    
#                    if running_ML == False:
#                        if dimensions == 1:
#                            print('Gen %1d Performance %3.3f, Distance = %3.1f, Velocity = %3.3f' % (I, best_performance[0], best_performance[1][0], best_performance[1][1]))
#                        elif dimensions == 2:
#                            print('Gen %1d Performance %3.3f, Distance_x = %3.1f, Velocity_x = %3.1f, Distance_y = %3.3f, Velocity_y = %3.1f' % (I, best_performance[0], best_performance[1][0], best_performance[1][1], best_performance[1][2], best_performance[1][3]))
#                     
#                    break
                
#            elif np.array_equal(pop['actions'][0,:,0]*pop['actions'][0,:,1],old_best):
#                
#                stall_count += 1
#                if stall_count >= 10:
#                    generation_nums[-1] += 1
#                    stall_count = 0
#                    print('Mutation Chance Increased',generation_nums[-1])
#            else:
#                stall_count = 0
#                generation_nums[-1] = orig_mutation_chance
#
#                
#            old_best = np.copy(pop['actions'][0,:,0]*pop['actions'][0,:,1])
                    
                
        elapsed_time = time.time() - start_time
#        print(elapsed_time)
        gen_count[0,KK] = I
        gen_count[1,KK] = elapsed_time
    gen_count_stats = np.zeros((3))
    gen_count_stats[:2] = np.median(gen_count,1)
    gen_count_stats[2] = np.sqrt(np.var(gen_count[0,:]))
    
    return pop, gen_count_stats

def set_goal(dimensions):
    
    if dimensions == 1:
        x_final = 100
        y_final = 0
        u_final = 0
        v_final = 0
    elif dimensions == 2:
        x_final = 0
        y_final = 0
        u_final = 0
        v_final = 5
    elif dimensions == 3:
        x_final = 0
        y_final = 0
        u_final = 0
        v_final = 5
        
    
    return np.array([x_final, u_final, y_final, v_final])

def set_parameters(dimensions, goal):
    
    dt = 1.0
    ideal_time_perc = 0.8
    t_steps = 20
    pop_size = 2000
    max_gen = 2000
    
    vel_max, acc = get_acceleration(dimensions, goal, dt, t_steps, ideal_time_perc)
    
    x_weight = 0.5
    u_weight = 0.5
    t_weight = 0.0
    
    #0.00153374 0.00153374 0.99693252
    
    elite_weight = 0.001
    lucky_weight = 0.001
    children_weight = 0.38
    mutation_weight = 0.51
    mutation_chance = 0.031
    
    GA_parameters = np.array([int(t_steps), int(pop_size), int(max_gen)])
    parameters = np.array([vel_max, acc, dt, ideal_time_perc])
    
    result_weights = (np.array([x_weight,u_weight,t_weight])
            /(x_weight+u_weight+t_weight))
    
    generation_weights = (np.concatenate(((np.array([elite_weight,lucky_weight,children_weight,mutation_weight])
    /(elite_weight+lucky_weight+children_weight+mutation_weight))
                ,np.array([mutation_chance])),axis=0))

    return GA_parameters, parameters, result_weights, generation_weights



if __name__ == "__main__":
    
    dimensions = 2
    samples = 1
    steps = 10
    deg_steps = 1
    
    goal = set_goal(dimensions)
        
    GA_parameters, parameters, result_weights, generation_weights = set_parameters(dimensions,goal)
    
    optimize_result_weights = False
    optimize_generation_weights = False    
    
    if optimize_result_weights == True:
        
        scored, new_weights = run_GA_result(result_weights,dimensions, goal, GA_parameters, parameters, generation_weights, samples, steps, deg_steps)
        
    elif optimize_generation_weights == True:
        
        scored, new_weights = run_GA_generation(generation_weights, dimensions, goal, GA_parameters, parameters, result_weights, samples, steps, deg_steps)
        
    else:
        pop, gen_count_stats  = run_GA(dimensions, goal, GA_parameters, parameters, result_weights, generation_weights, samples, steps, deg_steps)
        print(gen_count_stats[0])
    