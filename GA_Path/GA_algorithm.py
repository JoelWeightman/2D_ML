# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:50:44 2020

@author: jlwei
"""

import numpy as np
  
class GA_alg:
    
    def __init__(self,dimensions):
        
        self.t_steps = 100
        self.pop_size = 50
        self.max_gen = 100
        self.max_plateau = 10
        self.samples = 1
        
        self.deg_steps = 360
        self.vel_steps = 1
          
        self.r_weight = 0.1
        self.r_vel_weight = 0.1
        
        if dimensions == 1:
            self.theta_weight = 0.0
            self.theta_vel_weight = 0.0
            self.phi_weight = 0.0
            self.phi_vel_weight = 0.0
        elif dimensions == 2:
            self.theta_weight = 0.0
            self.theta_vel_weight = 0.1
            self.phi_weight = 0.0
            self.phi_vel_weight = 0.0
        else:
            self.theta_weight = 0.0
            self.theta_vel_weight = 0.1
            self.phi_weight = 0.1
            self.phi_vel_weight = 0.1
            
        self.t_weight = 0.0
        
        self.result_weights = (np.array([self.r_weight,self.r_vel_weight,self.theta_weight,
                            self.theta_vel_weight,self.phi_weight,self.phi_vel_weight,self.t_weight])
                            /(self.r_weight+self.r_vel_weight+self.theta_weight+self.theta_vel_weight
                              +self.phi_weight+self.phi_vel_weight+self.t_weight))
        
        self.radial_scale = 1e6
        self.radial_vel_scale = 100
        self.angle_scale = 0.01
        self.angle_vel_scale = 100/1e6*np.pi/180
        
        self.scales = np.array([self.radial_scale,self.radial_vel_scale,self.angle_scale,self.angle_vel_scale])
            
        self.elite_rollover = 2
        self.mutation_chance = 0.05
        self.parents_percentage = 0.75
        
        
    def generate_children(self,pop,dimensions):
        
        ## Select parents randomly
        # multiply previous generation by random number (0,1)
        # Pick 75% highest parents as seeders for next generation
        # Randomly pair up parents
        # each parent makes two children with random split point
        # For each child, randomly mutate genes using mutate_generation
        # score new children
        # Select N-2 parents
        # Keep 2 best of previous generation
        mutation_chance = self.mutation_chance
        parent_num = int(np.floor((self.parents_percentage)*self.pop_size/2)*2)
        
        weighted_parent_score = pop.score*np.random.rand(np.shape(pop.score)[0])
        
        parents_id = np.argsort(weighted_parent_score)[::-1][:parent_num]
        np.random.shuffle(parents_id)
        parents_id = np.tile(parents_id,2)
        
        split_points = np.random.randint(1,np.shape(pop.actions)[0]-1,int(parent_num))
        
        children_actions = np.zeros((np.shape(pop.actions[:,:,parents_id])))
        
        for i in range(int(parent_num)):
            j = 2*i
            children_actions[:,:,j] = np.concatenate((pop.actions[:split_points[i],:,parents_id[j]],pop.actions[split_points[i]:,:,parents_id[j+1]]),axis = 0)
            children_actions[:,:,j+1] = np.concatenate((pop.actions[split_points[i]:,:,parents_id[j]],pop.actions[:split_points[i],:,parents_id[j+1]]),axis = 0)

        children_actions = self.mutate_generation(children_actions,mutation_chance,dimensions)
        
        
        return children_actions
        
    
    def mutate_generation(self,actions,mutation_chance,dimensions):
        
        # given mutation rate (say 0.02)
        # Generate random number for each gene between (0,1)
        # for any number < mutation rate, randomly mutate that gene.
        
        mutation_field = np.random.rand(np.shape(actions)[0],np.shape(actions)[1],np.shape(actions)[2])
        
        potential_actions = np.zeros(np.shape(actions))
        potential_actions[:,0,:] = np.random.randint(0, self.vel_steps+1, size = np.shape(actions[:,0,:]))/self.vel_steps
        potential_actions[:,1,:] = np.random.randint(0, self.deg_steps, size = np.shape(actions[:,1,:]))*360/self.deg_steps*(np.pi/180)
        potential_actions[:,2,:] = np.random.randint(0, self.deg_steps+1, size = np.shape(actions[:,2,:]))*180/(self.deg_steps)*(np.pi/180)
    
        if dimensions == 1:
            mutation_field[:,1:,:] = 1
            actions[mutation_field<mutation_chance] = potential_actions[mutation_field<mutation_chance]
        elif dimensions == 2:
            mutation_field[:,2:,:] = 1
            
        actions[mutation_field<mutation_chance] = potential_actions[mutation_field<mutation_chance]
        
        return actions
        
    def generate_next_generation(self,pop):
        
        previous_id = np.argsort(pop.score)[::-1]
        children_id = np.argsort(pop.children_score)[::-1]
        
        elite_actions = pop.actions[:,:,previous_id[:2]]
        children_actions = pop.children_actions[:,:,children_id[:self.pop_size-2]]
        
        new_actions = np.concatenate((elite_actions,children_actions),axis=2)
        new_result = np.concatenate((pop.result[previous_id[:2],:],pop.children_result[children_id[:self.pop_size-2],:]),axis=0)
        new_score = np.concatenate((pop.score[previous_id[:2]],pop.children_score[children_id[:self.pop_size-2]]),axis=0)
        
        pop.actions = new_actions
        pop.result = new_result
        pop.score = new_score
        
        return pop
        
        
        
    def score_population(self, pop, result):
        
        scales = self.scales
        goal = pop.final_conditions
        result_weights = self.result_weights
        
        radial_val = np.log10(np.abs(result[:,0] - goal[0])/scales[0]+1.0)
        radial_vel_val = np.log10(np.abs(result[:,1] - goal[1])/scales[1] + 1.0)
                
        theta_angle_val = np.log10(np.abs(result[:,2] - goal[2])/scales[2] + 1.0)
        theta_angle_vel_val = np.log10(np.abs(result[:,3] - goal[3])/scales[3] + 1.0)
        
        phi_angle_val = np.log10(np.abs(result[:,4] - goal[4])/scales[2] + 1.0)
        phi_angle_vel_val = np.log10(np.abs(result[:,5] - goal[5])/scales[3] + 1.0)
        
        radial_val[radial_val > 1] = 1
        radial_vel_val[radial_vel_val > 1] = 1
        theta_angle_val[theta_angle_val > 1] = 1
        theta_angle_vel_val[theta_angle_vel_val > 1] = 1
        phi_angle_val[phi_angle_val > 1] = 1
        phi_angle_vel_val[phi_angle_vel_val > 1] = 1
        
        
        score = (
            (result_weights[0] - result_weights[0]*radial_val) + 
            (result_weights[1] - result_weights[1]*radial_vel_val) + 
            (result_weights[2] - result_weights[2]*theta_angle_val) +
            (result_weights[3] - result_weights[3]*theta_angle_vel_val) +
            (result_weights[4] - result_weights[4]*phi_angle_val) +
            (result_weights[5] - result_weights[5]*phi_angle_vel_val))

        score /= np.sum(result_weights)
        
        return score
    
        
        
        
       