# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:51:23 2020

@author: jlwei
"""
import numpy as np
from scipy.integrate import odeint

class Path(object):
    """The class defines the path object for motion""" 
    
    def __init__(self):
        
        self.path_method = 'orbital'
        self.path_method = 'linear'
               
    def run_initialisation(self,constants):
        
        self.get_constant_conditions(constants)
        self.get_init_conditions(constants)
        self.get_final_conditions(constants)
    

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
            
        self.initial_conditions = np.array([r_init, r_dot_init, np.deg2rad(theta_init), (theta_dot_init), 
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
        
        self.final_conditions = np.array([r_final, r_dot_final, np.deg2rad(theta_final), (theta_dot_final), 
                     np.deg2rad(phi_final), (phi_dot_final)])
    
    def get_constant_conditions(self,constants):
    
        R_E = constants.R_E
        G = constants.G
        M_E = constants.M_E
        dimensions = constants.dimensions
        
        g_0 = G*M_E/R_E**2
        
        if dimensions == 1:
            r_acc_const = 0
            theta_acc_const = 0
            phi_acc_const = 0
        elif dimensions == 2:
            r_acc_const = -g_0
            theta_acc_const = 0
            phi_acc_const = 0
        elif dimensions == 3:
            r_acc_const = -g_0
            theta_acc_const = 0
            phi_acc_const = 0
            
        self.constant_conditions = np.array([r_acc_const, (theta_acc_const), (phi_acc_const)])
    
    def initialise_population(self, GA_values):
        
        t_steps = GA_values.t_steps
        pop_size = GA_values.pop_size
        
        self.actions = np.zeros((t_steps,3,pop_size))

    def initialise_values(self, constants, GA_values):

        actions = self.actions
        
        vel_steps = GA_values.vel_steps
        deg_steps = GA_values.deg_steps
        dimensions = constants.dimensions
        
        if dimensions == 1:
            actions[:,0,:] = np.random.randint(-vel_steps, vel_steps+1, size = np.shape(actions[:,0,:]))/vel_steps
            actions[:,1,:] = 0*(np.pi/180)
            actions[:,2,:] = 90*(np.pi/180)
            
        elif dimensions == 2:
            actions[:,0,:] = np.random.randint(0, vel_steps+1, size = np.shape(actions[:,0,:]))/vel_steps
            actions[:,1,:] = np.random.randint(0, deg_steps, size = np.shape(actions[:,1,:]))*360/deg_steps*(np.pi/180)
            actions[:,2,:] = 90*(np.pi/180)
            
        elif dimensions == 3:
            actions[:,0,:] = np.random.randint(0, vel_steps+1, size = np.shape(actions[:,0,:]))/vel_steps
            actions[:,1,:] = np.random.randint(0, deg_steps, size = np.shape(actions[:,1,:]))*360/deg_steps*(np.pi/180)
            actions[:,2,:] = np.random.randint(0, deg_steps+1, size = np.shape(actions[:,2,:]))*180/(deg_steps)*(np.pi/180)
        
        self.actions = actions
    
    def calculate_path(self,actions,constants):
        
        acc = constants.acc
        dt = constants.dt
        init_conds = self.initial_conditions
        const_conds = self.constant_conditions
        
        d_x = np.zeros((np.shape(actions)[2],np.shape(actions)[0],1))
        d_y = np.zeros((np.shape(actions)[2],np.shape(actions)[0],1))
        d_z = np.zeros((np.shape(actions)[2],np.shape(actions)[0],1))
        
        v_x = np.zeros((np.shape(actions)[2],np.shape(actions)[0],1))
        v_y = np.zeros((np.shape(actions)[2],np.shape(actions)[0],1))
        v_z = np.zeros((np.shape(actions)[2],np.shape(actions)[0],1))
        
        r_cur = np.zeros((np.shape(actions)[2],1))
        theta_cur = np.zeros((np.shape(actions)[2],1))
        phi_cur = np.zeros((np.shape(actions)[2],1))
        
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
        
        for i in range(np.shape(actions)[0]-1):
            a_x = acc*actions[i,0,:]*np.cos(actions[i,1,:])*np.sin(actions[i,2,:]) + const_conds[0]*np.cos(theta_cur[:,0])*np.sin(phi_cur[:,0])
            a_y = acc*actions[i,0,:]*np.sin(actions[i,1,:])*np.sin(actions[i,2,:]) + const_conds[0]*np.sin(theta_cur[:,0])*np.sin(phi_cur[:,0])
            a_z = acc*actions[i,0,:]*np.cos(actions[i,2,:]) + const_conds[0]*np.cos(phi_cur[:,0])
            
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
        
        result = np.concatenate((r_cur, r_dot, theta_cur, theta_dot, phi_cur, phi_dot), axis = 1)
        
        return result
    
    def velocity_distance_3d_odeint(self, variables, t, actions, dt, acc, const_conds, R_E):
    
        d_x, d_y, d_z, v_x, v_y, v_z = variables
        
        r_cur = np.sqrt(d_x**2+d_y**2+d_z**2)
        theta_cur = np.arctan2(d_y,d_x)
        phi_cur = np.arctan2(np.sqrt(d_x**2+d_y**2),d_z)
    
        count = int(np.floor((t)/dt))

        dv_x_dt = acc*actions[count,0]*np.cos(actions[count,1])*np.sin(actions[count,2]) + const_conds[0]*np.cos(theta_cur)*np.sin(phi_cur)*(R_E/r_cur)**2
        dv_y_dt = acc*actions[count,0]*np.sin(actions[count,1])*np.sin(actions[count,2]) + const_conds[0]*np.sin(theta_cur)*np.sin(phi_cur)*(R_E/r_cur)**2
        dv_z_dt = acc*actions[count,0]*np.cos(actions[count,2]) + const_conds[0]*np.cos(phi_cur)*(R_E/r_cur)**2
        
        dd_x_dt = v_x
        dd_y_dt = v_y
        dd_z_dt = v_z
        
        return dd_x_dt, dd_y_dt, dd_z_dt, dv_x_dt, dv_y_dt, dv_z_dt
#    
    def calculate_path_odeint(self,actions,constants):
        
        init_conds = self.initial_conditions
        const_conds = self.constant_conditions
        R_E = constants.R_E
        acc = constants.acc
        dt = constants.dt
            
        d_x = init_conds[0]*np.cos(init_conds[2])*np.sin(init_conds[4])
        d_y = init_conds[0]*np.sin(init_conds[2])*np.sin(init_conds[4])
        d_z = init_conds[0]*np.cos(init_conds[4])
        
        v_x = (init_conds[1]*np.cos(init_conds[2])*np.sin(init_conds[4]) - 
            init_conds[0]*init_conds[3]*np.sin(init_conds[2])*np.sin(init_conds[4]) + 
            init_conds[0]*init_conds[5]*np.cos(init_conds[2])*np.cos(init_conds[4]))
        v_y = (init_conds[1]*np.sin(init_conds[2])*np.sin(init_conds[4]) + 
            init_conds[0]*init_conds[3]*np.cos(init_conds[2])*np.sin(init_conds[4]) + 
            init_conds[0]*init_conds[5]*np.sin(init_conds[2])*np.cos(init_conds[4]))
        v_z = (init_conds[1]*np.cos(init_conds[4]) - 
            init_conds[0]*init_conds[5]*np.sin(init_conds[4]))
        
        variables = np.array([d_x, d_y, d_z, v_x, v_y, v_z])
        
        r_dot = np.zeros((np.shape(actions)[2],1))
        r_cur = np.zeros((np.shape(actions)[2],1))
        theta_dot = np.zeros((np.shape(actions)[2],1))
        theta_cur = np.zeros((np.shape(actions)[2],1))
        phi_dot = np.zeros((np.shape(actions)[2],1))
        phi_cur = np.zeros((np.shape(actions)[2],1))
        
        t = np.linspace(0,dt*np.shape(actions)[0]-dt,np.shape(actions)[0])
        
        for pop_index in range(np.shape(actions)[2]):
            trajectory = odeint(self.velocity_distance_3d_odeint, variables, t, args=(actions[:,:,pop_index], dt, acc, const_conds, R_E), tcrit=t)

            [d_x, d_y, d_z, v_x, v_y, v_z] = trajectory[-1,:]
            
            r_cur[pop_index] = np.sqrt(d_x**2+d_y**2+d_z**2)
            theta_cur[pop_index] = np.arctan2(d_y,d_x)
            phi_cur[pop_index] = np.arctan2(np.sqrt(d_x**2+d_y**2),d_z)
        
            r_dot[pop_index] = (d_x*v_x + d_y*v_y + d_z*v_z)/r_cur[pop_index]
            theta_dot[pop_index] = (d_x*v_y - d_y*v_x)/(d_x**2)*np.cos(theta_cur[pop_index])**2
            phi_dot[pop_index] = ((d_x*d_z*v_x-
                    d_x**2*v_z+d_y*(d_z*v_y-
                    d_y*v_z))/(d_z**2*np.sqrt(d_x**2+d_y**2))*np.cos(phi_cur[pop_index])**2)
        
        result = np.concatenate((r_cur, r_dot, theta_cur, theta_dot, phi_cur, phi_dot), axis = 1)
        
        return result
    