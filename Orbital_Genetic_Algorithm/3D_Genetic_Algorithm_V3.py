"""
Created on Thu Jun 20 12:57:17 2019

@author: jlwei
"""

"""
    initial conditions done
    constant conditions (gravity) done
    change to 3D (r,theta,phi)
    
"""

import numpy as np
import time
import scipy.optimize as optim
import random as rdm
import matplotlib.pyplot as plt
import scipy.integrate as integ

def plot_actions(pop, acc, t_steps, dt, dimensions, init_conds, const_conds):
    
    v_x = np.zeros((np.shape(pop['actions'])[0],np.shape(pop['actions'])[1],1))
    v_y = np.zeros((np.shape(pop['actions'])[0],np.shape(pop['actions'])[1],1))
    
    v_x[:,1:t_steps,0] = np.cumsum(np.multiply(pop['actions'][:,:-1,0],(acc*np.cos(pop['actions'][:,:-1,1])*dt))+const_conds[0]*dt, axis = 1)+init_conds[1]
    v_y[:,1:t_steps,0] = np.cumsum(np.multiply(pop['actions'][:,:-1,0],(acc*np.sin(pop['actions'][:,:-1,1])*dt))+const_conds[1]*dt, axis = 1)+init_conds[3]
    
    v_x[:,0,0] = init_conds[1]
    v_y[:,0,0] = init_conds[3]
    
    d_x = integ.cumtrapz(v_x, dx = dt, axis = 1, initial = 0)+init_conds[0]
    d_y = integ.cumtrapz(v_y, dx = dt, axis = 1, initial = 0)+init_conds[2]
    
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

def get_acceleration(mass, thrust):
 
    u_acc = thrust/mass

    return u_acc          

def initialise_population(dimensions, t_steps, pop_size, vel_steps, deg_steps):
    
    pop = dict({'actions': np.zeros((pop_size,t_steps,dimensions)), 'results':np.zeros((pop_size,dimensions*2)), 'score':np.zeros((pop_size))})

    pop['actions'] = initialise_values(dimensions,pop['actions'], vel_steps, deg_steps)

    return pop

def initialise_values(dimensions, actions, vel_steps, deg_steps):

    if dimensions == 1:
        actions[:,:,0] = np.random.randint(-vel_steps, vel_steps+1, size = np.shape(actions[:,:,0]))/vel_steps
    elif dimensions == 2:
        actions[:,:,0] = np.random.randint(0, vel_steps+1, size = np.shape(actions[:,:,0]))/vel_steps
        actions[:,:,1] = np.random.randint(0, deg_steps, size = np.shape(actions[:,:,1]))*360/deg_steps*(np.pi/180)
    elif dimensions == 3:
        actions[:,:,0] = np.random.randint(0, vel_steps+1, size = np.shape(actions[:,:,0]))/vel_steps
        actions[:,:,1] = np.random.randint(0, deg_steps, size = np.shape(actions[:,:,1]))*360/deg_steps*(np.pi/180)
        actions[:,:,2] = np.random.randint(0, deg_steps+1, size = np.shape(actions[:,:,1]))*180/(deg_steps)*(np.pi/180)
    
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

    return generation_nums

def population_results(dimensions, pop, dt, acc, t_steps, init_conds, const_conds):
    
    if dimensions == 1:
                    
        r_dot, r = velocity_distance_1d(pop,dt,acc,t_steps,init_conds,const_conds)  
        
        pop['results'] = np.concatenate((r, r_dot), axis = 1)
       
    elif dimensions == 2:
    
        r_dot, theta_dot, r, theta = velocity_distance_2d(pop,dt,acc,t_steps,init_conds,const_conds)
        
        pop['results'] = np.concatenate((r, r_dot, theta, theta_dot), axis = 1)

    elif dimensions == 3:
        
        r_dot, theta_dot, phi_dot, r, theta, phi = velocity_distance_3d(pop,dt,acc,t_steps,init_conds,const_conds)
        
        pop['results'] = np.concatenate((r, r_dot, theta, theta_dot, phi, phi_dot), axis = 1)
        
    return pop

def velocity_distance_1d(pop, dt, acc, t_steps, init_conds, const_conds):
    
    r_dot = np.zeros((np.shape(pop['actions'])[0],np.shape(pop['actions'])[1],1))

    r_dot[:,1:t_steps,0] = np.cumsum(np.multiply(pop['actions'][:,:-1,0],(acc*dt))+const_conds[0]*dt, axis = 1)+init_conds[1]
    
    r_dot[:,0,0] = init_conds[1]
    
    r = np.trapz(r_dot, dx = dt, axis = 1)+init_conds[0]
    
    return r_dot[:,-1], r
    

def velocity_distance_2d(pop, dt, acc, t_steps, init_conds, const_conds):
    
    d_x = np.zeros((np.shape(pop['actions'])[0],np.shape(pop['actions'])[1],1))
    d_y = np.zeros((np.shape(pop['actions'])[0],np.shape(pop['actions'])[1],1))
   
    v_x = np.zeros((np.shape(pop['actions'])[0],np.shape(pop['actions'])[1],1))
    v_y = np.zeros((np.shape(pop['actions'])[0],np.shape(pop['actions'])[1],1))
    
    r_cur = np.zeros((np.shape(pop['actions'])[0],1))
    theta_cur = np.zeros((np.shape(pop['actions'])[0],1))
    
    d_x[:,0,0] = init_conds[0]*np.cos(init_conds[2])
    d_y[:,0,0] = init_conds[0]*np.sin(init_conds[2])
    
    v_x[:,0,0] = (init_conds[1]*np.cos(init_conds[2]) - 
        init_conds[0]*init_conds[3]*np.sin(init_conds[2]))
    v_y[:,0,0] = (init_conds[1]*np.sin(init_conds[2]) + 
        init_conds[0]*init_conds[3]*np.cos(init_conds[2]))
    
    r_cur[:,0] = init_conds[0]
    theta_cur[:,0] = init_conds[2]

    for i in range(np.shape(pop['actions'])[1]-1):
        a_x = acc*pop['actions'][:,i,0]*np.cos(pop['actions'][:,i,1]) + const_conds[0]*np.cos(theta_cur[:,0])
        a_y = acc*pop['actions'][:,i,0]*np.sin(pop['actions'][:,i,1])
        
        v_x[:,i+1,0] = a_x*dt + v_x[:,i,0]
        v_y[:,i+1,0] = a_y*dt + v_y[:,i,0]
        
        d_x[:,i+1,0] = v_x[:,i,0]*dt + d_x[:,i,0]
        d_y[:,i+1,0] = v_y[:,i,0]*dt + d_y[:,i,0]
        
        r_cur = np.sqrt(d_x[:,i+1]**2+d_y[:,i+1]**2)
        theta_cur = np.arctan2(d_y[:,i+1],d_x[:,i+1])
   
    
    r_dot = (d_x[:,-1]*v_x[:,-1] + d_y[:,-1]*v_y[:,-1])/r_cur
    theta_dot = (d_x[:,-1]*v_y[:,-1] - d_y[:,-1]*v_x[:,-1])/(d_x[:,-1]**2)*np.cos(theta_cur)**2
  
    return r_dot, theta_dot, r_cur, theta_cur
             

def velocity_distance_3d(pop, dt, acc, t_steps, init_conds, const_conds):
    
    d_x = np.zeros((np.shape(pop['actions'])[0],np.shape(pop['actions'])[1],1))
    d_y = np.zeros((np.shape(pop['actions'])[0],np.shape(pop['actions'])[1],1))
    d_z = np.zeros((np.shape(pop['actions'])[0],np.shape(pop['actions'])[1],1))
    
    v_x = np.zeros((np.shape(pop['actions'])[0],np.shape(pop['actions'])[1],1))
    v_y = np.zeros((np.shape(pop['actions'])[0],np.shape(pop['actions'])[1],1))
    v_z = np.zeros((np.shape(pop['actions'])[0],np.shape(pop['actions'])[1],1))
    
    r_cur = np.zeros((np.shape(pop['actions'])[0],1))
    theta_cur = np.zeros((np.shape(pop['actions'])[0],1))
    phi_cur = np.zeros((np.shape(pop['actions'])[0],1))
    
    d_x[:,0,0] = init_conds[0]*np.cos(init_conds[2])*np.sin(init_conds[4])
    d_y[:,0,0] = init_conds[0]*np.sin(init_conds[2])*np.sin(init_conds[4])
    d_z[:,0,0] = init_conds[0]*np.cos(init_conds[4])
    
    v_x[:,0,0] = (init_conds[1]*np.cos(init_conds[2])*np.sin(init_conds[4]) - 
        init_conds[0]*init_conds[3]*np.sin(init_conds[2])*np.sin(init_conds[4]) + 
        init_conds[0]*init_conds[5]*np.cos(init_conds[2])*np.cos(init_conds[4]))
    v_y[:,0,0] = (init_conds[1]*np.sin(init_conds[2])*np.sin(init_conds[4]) + 
        init_conds[0]*init_conds[3]*np.cos(init_conds[2])*np.sin(init_conds[4]) + 
        init_conds[0]*init_conds[5]*np.sin(init_conds[2])*np.cos(init_conds[4]))
    v_z[:,0,0] = (init_conds[1]*np.cos(init_conds[4]) - 
        init_conds[0]*init_conds[5]*np.sin(init_conds[4]))
    
    r_cur[:,0] = init_conds[0]
    theta_cur[:,0] = init_conds[2]
    phi_cur[:,0] = init_conds[4]
    
    for i in range(np.shape(pop['actions'])[1]-1):
        a_x = acc*pop['actions'][:,i,0]*np.cos(pop['actions'][:,i,1])*np.sin(pop['actions'][:,i,2]) + const_conds[0]*np.cos(theta_cur[:,0])*np.sin(phi_cur[:,0])
        a_y = acc*pop['actions'][:,i,0]*np.sin(pop['actions'][:,i,1])*np.sin(pop['actions'][:,i,2])
        a_z = acc*pop['actions'][:,i,0]*np.cos(pop['actions'][:,i,2])
        
        v_x[:,i+1,0] = a_x*dt + v_x[:,i,0]
        v_y[:,i+1,0] = a_y*dt + v_y[:,i,0]
        v_z[:,i+1,0] = a_z*dt + v_z[:,i,0]
        
        d_x[:,i+1,0] = v_x[:,i,0]*dt + d_x[:,i,0]
        d_y[:,i+1,0] = v_y[:,i,0]*dt + d_y[:,i,0]
        d_z[:,i+1,0] = v_z[:,i,0]*dt + d_z[:,i,0]
        
        r_cur = np.sqrt(d_x[:,i+1]**2+d_y[:,i+1]**2+d_z[:,i+1]**2)
        theta_cur = np.arctan2(d_y[:,i+1],d_x[:,i+1])
        phi_cur = np.arctan2(np.sqrt(d_x[:,i+1]**2+d_y[:,i+1]**2),d_z[:,i+1])
    
    r_dot = (d_x[:,-1]*v_x[:,-1] + d_y[:,-1]*v_y[:,-1] + d_z[:,-1]*v_z[:,-1])/r_cur
    theta_dot = (d_x[:,-1]*v_y[:,-1] - d_y[:,-1]*v_x[:,-1])/(d_x[:,-1]**2)*np.cos(theta_cur)**2
    phi_dot = ((d_x[:,-1]*d_z[:,-1]*v_x[:,-1]-
            d_x[:,-1]**2*v_z[:,-1]+d_y[:,-1]*(d_z[:,-1]*v_y[:,-1]-
            d_y[:,-1]*v_z[:,-1]))/(d_z[:,-1]**2*np.sqrt(d_x[:,-1]**2+d_y[:,-1]**2))*np.cos(phi_cur)**2)
    
    
    return r_dot, theta_dot, phi_dot, r_cur, theta_cur, phi_cur

def population_score(dimensions, pop, result_weights, goal, parameters):

    [acc, dt] = parameters
       
    radial_scale = 1
    radial_vel_scale = 0.1
    angle_scale = 1*np.pi/180
    angle_vel_scale = 0.1*np.pi/180
    
    pop['actions'][:,-1,:] = 0.0
    
    if dimensions == 1:
        radial_val = np.abs(pop['results'][:,0] - goal[0])/radial_scale
        radial_vel_val = np.abs(pop['results'][:,1] - goal[1])/radial_vel_scale
        
        pop['score'] = (
            (result_weights[0] - result_weights[0]*radial_val) + 
            (result_weights[1] - result_weights[1]*radial_vel_val))

        
    elif dimensions == 2:
        radial_val = np.abs(pop['results'][:,0] - goal[0])/radial_scale
        radial_vel_val = np.abs(pop['results'][:,1] - goal[1])/radial_vel_scale
        
        angle_val = np.abs(pop['results'][:,2] - goal[2])/angle_scale
        angle_vel_val = np.abs(pop['results'][:,3] - goal[3])/angle_vel_scale
        
        pop['score'] = (
            (result_weights[0] - result_weights[0]*radial_val) + 
            (result_weights[1] - result_weights[1]*radial_vel_val) + 
            (result_weights[2] - result_weights[2]*angle_val) +
            (result_weights[3] - result_weights[3]*angle_vel_val))
        
    elif dimensions == 3:
        radial_val = np.abs(pop['results'][:,0] - goal[0])/radial_scale
        radial_vel_val = np.abs(pop['results'][:,1] - goal[1])/radial_vel_scale
        
        theta_angle_val = np.abs(pop['results'][:,2] - goal[2])/angle_scale
        theta_angle_vel_val = np.abs(pop['results'][:,3] - goal[3])/angle_vel_scale
        phi_angle_val = np.abs(pop['results'][:,4] - goal[4])/angle_scale
        phi_angle_vel_val = np.abs(pop['results'][:,5] - goal[5])/angle_vel_scale
        
        pop['score'] = (
            (result_weights[0] - result_weights[0]*radial_val) + 
            (result_weights[1] - result_weights[1]*radial_vel_val) + 
            (result_weights[2] - result_weights[2]*theta_angle_val) +
            (result_weights[3] - result_weights[3]*theta_angle_vel_val) +
            (result_weights[4] - result_weights[4]*phi_angle_val) +
            (result_weights[5] - result_weights[5]*phi_angle_vel_val))

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
            if indices[2][i] == 0:
                if dimensions == 1:
                    mutated_actions[indices[0][i],indices[1][i],indices[2][i]] = np.random.randint(-vel_steps, vel_steps+1, size = 1)/vel_steps
                else:
                    mutated_actions[indices[0][i],indices[1][i],indices[2][i]] = np.random.randint(0, vel_steps+1, size = 1)/vel_steps
            elif indices[2][i] == 1:
                mutated_actions[indices[0][i],indices[1][i],indices[2][i]] = np.random.randint(0, deg_steps, size = 1)*360/deg_steps*(np.pi/180)
            elif indices[2][i] == 2:
                mutated_actions[indices[0][i],indices[1][i],indices[2][i]] = np.random.randint(0, deg_steps + 1, size = 1)*180/deg_steps*(np.pi/180)
    
    
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
        list_3 = np.array([0])#np.linspace(-0.5,0.5,step_size)/step_size**iteration+centre_values[2]
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
                    print(iteration,I,J,K,gen_count_stats[0],gen_count_stats[1])
                    scored[I,J,K] = gen_count_stats[0] + 2*gen_count_stats[1]
        
        min_val_loc = np.where(scored == np.min(scored))
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
                            print(iteration,I,J,K,M,N,gen_count_stats[0],gen_count_stats[1])
                            scored[I,J,K,M,N] = gen_count_stats[0] + 2*gen_count_stats[1]
        
        min_val_loc = np.where(scored == np.min(scored))
        centre_values = np.array([list_1[min_val_loc[0]],list_2[min_val_loc[1]],list_3[min_val_loc[2]],list_4[min_val_loc[3]],list_5[min_val_loc[4]]])
    
    return scored, centre_values

def run_GA(dimensions, goal, GA_parameters, parameters, result_weights, generation_weights, samples, vel_steps, deg_steps, running_ML = False):
    
    [t_steps, pop_size, max_gen] = GA_parameters
    [acc, dt] = parameters
    
    init_conds = initial_conditions(dimensions)
    const_conds = constant_conditions(dimensions)
    
    generation_nums = get_generation_nums(t_steps, pop_size, generation_weights)
    
    orig_mutation_chance = np.copy(generation_nums[-1])
    stall_count = 0

    theory_max = (np.sum(result_weights[:-1]) + result_weights[-1])/(np.sum(result_weights))
        
    thresh_perf = 1e-4
     
    gen_count = np.zeros((2,samples))
       
    for KK in range(samples):  # Count for samples
        start_time = time.time()
        pop = initialise_population(dimensions, t_steps, pop_size, vel_steps, deg_steps)
#        old_best = np.copy(pop['actions'][0,:,0]*pop['actions'][0,:,1])
        
        for I in range(max_gen):
            pop = population_results(dimensions, pop, dt, acc, t_steps, init_conds, const_conds)
            pop = population_score(dimensions, pop, result_weights, goal, parameters)
            pop, best_performance = pop_selection(pop, generation_nums, dimensions, vel_steps, deg_steps)

            if running_ML == False:
                if np.mod(I,max_gen/10) == 0 or I == 0:
                    print('t: %d, Pop: %d, Run: %1d, Max Perf: %3.3f' % (t_steps, pop_size, KK, theory_max))
    
                    if False:
                        plt.figure(0)
                        plt.clf()
                        if dimensions == 1:
                            CS = plt.imshow(pop['actions'][:,:,0], aspect='auto')
                        elif dimensions == 2:
                            CS = plt.imshow(pop['actions'][:,:,0]*pop['actions'][:,:,1], aspect='auto')
                        elif dimensions == 3:
                            CS = plt.imshow(pop['actions'][:,:,0]*pop['actions'][:,:,1], aspect='auto')
                        plt.colorbar(CS)
                        plt.pause(0.01)
                        
                    if False:
                        plot_actions(pop, acc, t_steps, dt, dimensions, init_conds, const_conds)
                        
                    if dimensions == 1:
                        print('Gen %1d Performance %3.3f, Distance = %3.1f, Velocity = %3.3f' % (I, best_performance[0], best_performance[1][0], best_performance[1][1]))
                    elif dimensions == 2:
                        print('Gen %1d Performance %3.3f, Radius = %3.1f, Radial Velocity = %3.1f, Angle = %3.3f, Angular Velocity = %3.1f' % (I, best_performance[0], best_performance[1][0], best_performance[1][1], np.rad2deg(best_performance[1][2]), np.rad2deg(best_performance[1][3])))
                    elif dimensions == 3:
                        print('Gen %1d Performance %3.3f, Radius = %3.1f, Radial Velocity = %3.1f, Theta Angle = %3.3f, Theta Angular Velocity = %3.1f, Phi Angle = %3.3f, Phi Angular Velocity = %3.1f' % (I, best_performance[0], best_performance[1][0], best_performance[1][1], np.rad2deg(best_performance[1][2]), np.rad2deg(best_performance[1][3]), np.rad2deg(best_performance[1][4]), np.rad2deg(best_performance[1][5])))

    
                if I == max_gen - 1:
                    print('t: %d, Pop: %d, Run: %1d, Max Perf: %3.3f' % (t_steps, pop_size, KK, theory_max))
                    if dimensions == 1:
                        print('Gen %1d Performance %3.3f, Distance = %3.1f, Velocity = %3.3f' % (I, best_performance[0], best_performance[1][0], best_performance[1][1]))
                    elif dimensions == 2:
                        print('Gen %1d Performance %3.3f, Radius = %3.1f, Radial Velocity = %3.1f, Angle = %3.3f, Angular Velocity = %3.1f' % (I, best_performance[0], best_performance[1][0], best_performance[1][1], np.rad2deg(best_performance[1][2]), np.rad2deg(best_performance[1][3])))
                    elif dimensions == 3:
                        print('Gen %1d Performance %3.3f, Radius = %3.1f, Radial Velocity = %3.1f, Theta Angle = %3.3f, Theta Angular Velocity = %3.1f, Phi Angle = %3.3f, Phi Angular Velocity = %3.1f' % (I, best_performance[0], best_performance[1][0], best_performance[1][1], np.rad2deg(best_performance[1][2]), np.rad2deg(best_performance[1][3]), np.rad2deg(best_performance[1][4]), np.rad2deg(best_performance[1][5])))


            if best_performance[0] >= theory_max-thresh_perf*theory_max:
                    
                    sorted_pop = np.argsort(pop['score'])[::-1]
                    
                    pop['results'] = pop['results'][sorted_pop]
                    pop['score'] = pop['score'][sorted_pop] 
                    
                    if running_ML == False:
                        if dimensions == 1:
                            print('Gen %1d Performance %3.3f, Distance = %3.1f, Velocity = %3.3f' % (I, best_performance[0], best_performance[1][0], best_performance[1][1]))
                        elif dimensions == 2:
                            print('Gen %1d Performance %3.3f, Radius = %3.1f, Radial Velocity = %3.1f, Angle = %3.3f, Angular Velocity = %3.1f' % (I, best_performance[0], best_performance[1][0], best_performance[1][1], np.rad2deg(best_performance[1][2]), np.rad2deg(best_performance[1][3])))
                        elif dimensions == 3:
                            print('Gen %1d Performance %3.3f, Radius = %3.1f, Radial Velocity = %3.1f, Theta Angle = %3.3f, Theta Angular Velocity = %3.1f, Phi Angle = %3.3f, Phi Angular Velocity = %3.1f' % (I, best_performance[0], best_performance[1][0], best_performance[1][1], np.rad2deg(best_performance[1][2]), np.rad2deg(best_performance[1][3]), np.rad2deg(best_performance[1][4]), np.rad2deg(best_performance[1][5])))

                    break
                
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
        r_final = 100
        theta_final = 0
        phi_final = 0
        r_dot_final = 0
        theta_dot_final = 0
        phi_dot_final = 0
    elif dimensions == 2:
        r_final = 100
        theta_final = 30
        phi_final = 0
        r_dot_final = 0
        theta_dot_final = 0
        phi_dot_final = 0
    elif dimensions == 3:
        r_final = 100
        theta_final = 30
        phi_final = 60
        r_dot_final = 0
        theta_dot_final = 0
        phi_dot_final = 0
    
    return np.array([r_final, r_dot_final, np.deg2rad(theta_final), np.deg2rad(theta_dot_final), 
                     np.deg2rad(phi_final), np.deg2rad(phi_dot_final)])

def initial_conditions(dimensions):
    
    if dimensions == 1:
        r_init = 0
        theta_init = 0
        phi_init = 90
        r_dot_init = 0
        theta_dot_init = 0
        phi_dot_init = 0
    elif dimensions == 2:
        r_init = 0
        theta_init = 0
        phi_init = 90
        r_dot_init = 0
        theta_dot_init = 0
        phi_dot_init = 0
    elif dimensions == 3:
        r_init = 0
        theta_init = 0
        phi_init = 90
        r_dot_init = 0
        theta_dot_init = 0
        phi_dot_init = 0
        
    return np.array([r_init, r_dot_init, np.deg2rad(theta_init), np.deg2rad(theta_dot_init), 
                     np.deg2rad(phi_init), np.deg2rad(phi_dot_init)])
    
def constant_conditions(dimensions):
    
    if dimensions == 1:
        r_acc_const = 0
        theta_acc_const = 0
        phi_acc_const = 0
    elif dimensions == 2:
        r_acc_const = 0
        theta_acc_const = 0
        phi_acc_const = 0
    elif dimensions == 3:
        r_acc_const = 0
        theta_acc_const = 0
        phi_acc_const = 0
        
    return np.array([r_acc_const, np.deg2rad(theta_acc_const), np.deg2rad(phi_acc_const)])

def set_parameters(dimensions, goal):
    
    dt = 1.0
    t_steps = 20
    pop_size = 50
    max_gen = 2000
    
    mass = 1
    thrust = 5
    
    acc = get_acceleration(mass, thrust)
    
    r_weight = 0.5
    r_vel_weight = 0.5
    theta_weight = 0.5
    theta_vel_weight = 0.5
    phi_weight = 0.5
    phi_vel_weight = 0.5
    t_weight = 0
        
    elite_weight = 0.001
    lucky_weight = 0.001
    children_weight = 0.75
    mutation_weight = 0.248
    mutation_chance = 0.025
    
    GA_parameters = np.array([int(t_steps), int(pop_size), int(max_gen)])
    parameters = np.array([acc, dt])
    
    if dimensions == 1:
        result_weights = (np.array([r_weight,r_vel_weight,t_weight])
                /(r_weight+r_vel_weight+t_weight))
    elif dimensions == 2:
        result_weights = (np.array([r_weight,r_vel_weight,theta_weight,theta_vel_weight,t_weight])
            /(r_weight+r_vel_weight+theta_weight+theta_vel_weight+t_weight))
    elif dimensions == 3:
        result_weights = (np.array([r_weight,r_vel_weight,theta_weight,theta_vel_weight,phi_weight,phi_vel_weight,t_weight])
            /(r_weight+r_vel_weight+theta_weight+theta_vel_weight+phi_weight+phi_vel_weight+t_weight))
    
    generation_weights = (np.concatenate(((np.array([elite_weight,lucky_weight,children_weight,mutation_weight])
    /(elite_weight+lucky_weight+children_weight+mutation_weight))
                ,np.array([mutation_chance])),axis=0))

    return GA_parameters, parameters, result_weights, generation_weights



if __name__ == "__main__":
    
    dimensions = 3
    samples = 1
    vel_steps = 4
    deg_steps = 12
    
    goal = set_goal(dimensions)
        
    GA_parameters, parameters, result_weights, generation_weights = set_parameters(dimensions,goal)
    
    optimize_result_weights = False
    optimize_generation_weights = False    
    
    if optimize_result_weights == True:
        
        scored, new_weights = run_GA_result(result_weights,dimensions, goal, GA_parameters, parameters, generation_weights, samples, steps, deg_steps)
        
    elif optimize_generation_weights == True:
        
        scored, new_weights = run_GA_generation(generation_weights, dimensions, goal, GA_parameters, parameters, result_weights, samples, steps, deg_steps)
        
    else:
        pop, gen_count_stats  = run_GA(dimensions, goal, GA_parameters, parameters, result_weights, generation_weights, samples, vel_steps, deg_steps)
        print(gen_count_stats[0])
    