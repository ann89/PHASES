#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 21:36:47 2018

@author: 
"""

import MDAnalysis
import MDAnalysis.lib.NeighborSearch as NS
import numpy as np
import sys
import multiprocessing as mp
from time import time
from numpy.linalg import norm
import scipy.linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

top  = 'membrane-start.pdb'
traj = '150_ns_membrane_whole-100ps.xtc'
side = "up" #sys.argv[1] # "up" for upper leaflet "down" for lower leaflet
skip = 140

u = MDAnalysis.Universe(top, traj)
n_frames = u.trajectory.n_frames
frames = np.arange(n_frames)[::skip]

vref = np.array([0.0,0.0,1.0])

def get_side_coordinates_and_box(u):
    """Assign lipids to leaflets, retrieve their coordinates, resIDs and the director order parameter u=3/2cosˆ2(theta)-1/2"""
    
    #u = MDAnalysis.Universe(top,traj)
    #u.trajectory[frame]

    x, y, z = u.trajectory.ts.triclinic_dimensions[0][0], u.trajectory.ts.triclinic_dimensions[1][1], u.trajectory.ts.triclinic_dimensions[2][2]
    box = np.array([x, y, z])
    
    ### Determining side of the bilayer CHOL belongs to in this frame
    #Lipid Residue names
    lipid1 ='DPPC'
    lipid2 ='DAPC'
    lipid3 ='CHL'
        
    lpd1_atoms = u.select_atoms('resname %s and name P'%lipid1)
    lpd2_atoms = u.select_atoms('resname %s and name P'%lipid2)
    lpd3_atoms = u.select_atoms('resname %s and name O2'%lipid3)
    num_lpd1 = lpd1_atoms.n_atoms
    num_lpd2 = lpd2_atoms.n_atoms
        # atoms in the upper leaflet as defined by insane.py or the CHARMM-GUI membrane builders
        # select cholesterol headgroups within 1.5 nm of lipid headgroups in the selected leaflet
        # this must be done because CHOL rapidly flip-flops between leaflets in the MARTINI model
        # so we must assign CHOL to each leaflet at every time step, and in large systems
        # with substantial membrane undulations, a simple cut-off in the z-axis just will not cut it
    if side == 'up':
        lpd1i = lpd1_atoms[:int((num_lpd1)/2)]
        lpd2i = lpd2_atoms[:int((num_lpd2)/2)]
        lipids = lpd1i + lpd2i
        ns_lipids = NS.AtomNeighborSearch(lpd3_atoms)
        lpd3i = ns_lipids.search(lipids,15.0)
    elif side == 'down':
        lpd1i = lpd1_atoms[int((num_lpd1)/2):]
        lpd2i = lpd2_atoms[int((num_lpd2)/2):]
        lipids = lpd1i + lpd2i
        ns_lipids = NS.AtomNeighborSearch(lpd3_atoms)
        lpd3i = ns_lipids.search(lipids,15.0)
    
    n_cells_per_unit_cell = 0.05
    n_cells_x = int(x * n_cells_per_unit_cell) + 1
    
    n_cells_y = int(y * n_cells_per_unit_cell) + 1
    
    # fill particles into cells
    lpd2_cell_idx = np.zeros((len(lpd2i.resnums),1))
    vect_DAPC = np.zeros((len(lpd2i.resnums),3))
    for i in range(len(lpd2i.resnums)):
        lpd2_cell_idx[i] = int(lpd2i.positions[i][0] * n_cells_per_unit_cell) + n_cells_x * int(lpd2i.positions[i][1] * n_cells_per_unit_cell)
         #calculate the tail-end vector for DAPC        
        resnum_DAPC = lpd2i.resnums[i]
        head_DAPC = u.select_atoms('resnum %i and (name C22)'%resnum_DAPC).center_of_geometry() 
        tail_DAPC = u.select_atoms('resnum %i and (name 0C22)'%resnum_DAPC).center_of_geometry()
        vect_DAPC[i, 0] = head_DAPC[0] - tail_DAPC[0]
        vect_DAPC[i, 1] = head_DAPC[1] - tail_DAPC[1]
        vect_DAPC[i, 2] = head_DAPC[2] - tail_DAPC[2] 
        vect_DAPC[i] /= norm(vect_DAPC[i])
           
        
    
    lpd1_cell_idx = np.zeros((len(lpd1i.resnums),1))
    vect_DPPC = np.zeros((len(lpd1i.resnums),3))
    for i in range(len(lpd1i.resnums)):
        lpd1_cell_idx[i] = int(int(lpd1i.positions[i][0] * n_cells_per_unit_cell) + n_cells_x * int(lpd1i.positions[i][1] * n_cells_per_unit_cell))
        #calculate the tail-end vector for DPPC        
        resnum_DPPC = lpd1i.resnums[i]
        head_DPPC = u.select_atoms('resnum %i and (name C22)'%resnum_DPPC).center_of_geometry() 
        tail_DPPC = u.select_atoms('resnum %i and (name 6C21)'%resnum_DPPC).center_of_geometry()
        vect_DPPC[i, 0] = head_DPPC[0] - tail_DPPC[0]
        vect_DPPC[i, 1] = head_DPPC[1] - tail_DPPC[1]
        vect_DPPC[i, 2] = head_DPPC[2] - tail_DPPC[2] 
        vect_DPPC[i] /= norm(vect_DPPC[i]) 
    
    
    
    lpd3_cell_idx = np.zeros((len(lpd3i.resnums),1))
    vect_CHL = np.zeros((len(lpd3i.resnums),3))
    for i in range(len(lpd3i.resnums)):
        lpd3_cell_idx[i] = int(lpd3i.positions[i][0] * n_cells_per_unit_cell) + n_cells_x * int(lpd3i.positions[i][1] * n_cells_per_unit_cell)
        
        #calculate the tail-end vector for CHL
        resnum = lpd3i.resnums[i]
        head_CHL = u.select_atoms('resnum %i and (name C1)'%resnum).center_of_geometry() 
        tail_CHL = u.select_atoms('resnum %i and (name C65)'%resnum).center_of_geometry()
        vect_CHL[i, 0] = head_CHL[0] - tail_CHL[0]
        vect_CHL[i, 1] = head_CHL[1] - tail_CHL[1]
        vect_CHL[i, 2] = head_CHL[2] - tail_CHL[2]
        vect_CHL[i] /= norm(vect_CHL[i])
       
    
    v_normal_cells = np.zeros((n_cells_x * n_cells_y,3))
    # go through the cells
    for idx in range(n_cells_x * n_cells_y):
        x_lpd1_in_cell = lpd1i.positions[(lpd1_cell_idx[:] == idx).reshape((len(lpd1_cell_idx[:]))), 0]
        y_lpd1_in_cell = lpd1i.positions[(lpd1_cell_idx[:] == idx).reshape((len(lpd1_cell_idx[:]))), 1]
        z_lpd1_in_cell = lpd1i.positions[(lpd1_cell_idx[:] == idx).reshape((len(lpd1_cell_idx[:]))), 2]
        
        x_lpd2_in_cell = lpd2i.positions[(lpd2_cell_idx[:] == idx).reshape((len(lpd2_cell_idx[:]))), 0]
        y_lpd2_in_cell = lpd2i.positions[(lpd2_cell_idx[:] == idx).reshape((len(lpd2_cell_idx[:]))), 1]
        z_lpd2_in_cell = lpd2i.positions[(lpd2_cell_idx[:] == idx).reshape((len(lpd2_cell_idx[:]))), 2]
        
        x_lpd3_in_cell = lpd3i.positions[(lpd3_cell_idx[:] == idx).reshape((len(lpd3_cell_idx[:]))), 0]
        y_lpd3_in_cell = lpd3i.positions[(lpd3_cell_idx[:] == idx).reshape((len(lpd3_cell_idx[:]))), 1]
        z_lpd3_in_cell = lpd3i.positions[(lpd3_cell_idx[:] == idx).reshape((len(lpd3_cell_idx[:]))), 2]
        
        x_lpdall = np.concatenate((x_lpd1_in_cell, x_lpd2_in_cell, x_lpd3_in_cell), axis=0)
        y_lpdall = np.concatenate((y_lpd1_in_cell, y_lpd2_in_cell, y_lpd3_in_cell), axis=0)
        z_lpdall = np.concatenate((z_lpd1_in_cell, z_lpd2_in_cell, z_lpd3_in_cell), axis=0)

        #print(len(x_lpd1_in_cell), len(x_lpd2_in_cell), len(x_lpd3_in_cell))
        print(x_lpdall.shape)
        # best-fit linear plane
        A = np.c_[x_lpdall, y_lpdall, np.ones(len(y_lpdall))]
        C,_,_,_ = scipy.linalg.lstsq(A, z_lpdall)    # coefficients
        
        # normal to plane [c0, c1, -1]
        v_normal = np.array([C[0], C[1], -1.0])
        v_normal_abs = np.sqrt(v_normal[0]**2 + v_normal[1]**2 + v_normal[2]**2)
        v_normal = v_normal / v_normal_abs
        v_normal_cells[idx] = v_normal
    
    print(v_normal_cells)
    lpd1_coords = np.zeros((len(lpd1i.resnums),3))
    lpd1_res = np.zeros((len(lpd1i.resnums),1))
    lpd1_u = np.zeros((len(lpd1i.resnums),1))
    for i in np.arange(len(lpd1i.resnums)):
        vref = v_normal_cells[int(lpd1_cell_idx[i])]
        resnum = lpd1i.resnums[i]
        theta = np.arccos(np.dot(vect_DPPC[i], vref))
        u_ord = 1.5*((pow(np.cos(theta), 2)))-0.5
        group = u.select_atoms('resnum %i'%resnum)
        group_cog = group.center_of_geometry()
        lpd1_coords[i] = group_cog
        lpd1_res[i] = resnum
        lpd1_u[i] = u_ord
    
    lpd2_coords = np.zeros((len(lpd2i.resnums),3))
    lpd2_res = np.zeros((len(lpd2i.resnums),1))
    lpd2_u = np.zeros((len(lpd2i.resnums),1))
    for i in np.arange(len(lpd2i.resnums)):
        vref = v_normal_cells[int(lpd2_cell_idx[i])]
        resnum = lpd2i.resnums[i]
        theta = np.arccos(np.dot(vect_DAPC[i], vref))
        u_ord = 1.5*((pow(np.cos(theta), 2)))-0.5
        group = u.select_atoms('resnum %i'%resnum)
        group_cog = group.center_of_geometry()
        lpd2_coords[i] = group_cog
        lpd2_res[i] = resnum
        lpd2_u[i] = u_ord   
    
      
    lpd3_coords = np.zeros((len(lpd3i.resnums),3))
    lpd3_res = np.zeros((len(lpd3i.resnums),1))
    lpd3_u = np.zeros((len(lpd3i.resnums),1))
    for i in np.arange(len(lpd3i.resnums)):
        vref = v_normal_cells[int(lpd3_cell_idx[i])]
        resnum = lpd3i.resnums[i]
        theta = np.arccos(np.dot(vect_CHL[i], vref))
        u_ord = 1.5*((pow(np.cos(theta), 2)))-0.5
        group = u.select_atoms('resnum %i'%resnum)
        group_cog = group.center_of_geometry()
        lpd3_coords[i] = group_cog
        lpd3_res[i] = resnum
        lpd3_u[i] = u_ord

    
    lpd_coords = np.vstack((lpd1_coords,lpd1_coords,lpd2_coords,lpd2_coords,lpd3_coords)) #append
    lpd_resids = np.vstack((lpd1_res,lpd1_res, lpd2_res, lpd2_res, lpd3_res)) 
    lpd_us =np.vstack((lpd1_u,lpd1_u, lpd2_u,lpd2_u, lpd3_u))
    lpd_coords = lpd_coords.astype('float32')
    lpd_resids = lpd_resids.astype('float32')
    lpd_us = lpd_us.astype('float32')
     
     
    return lpd_coords,lpd_resids,lpd_us, box        
            


    #for i in np.arange(len(lpd2i.resnums)):
        
    #for i in np.arange(len(lpd3i.resnums)):

    #return lpd_coords,lpd_resids, u_all, box
    '''
    print(theta_all)
    print(v_normal, np.sqrt(v_normal[0]**2 + v_normal[1]**2 + v_normal[2]**2))
    #np.meshgrid(np.arange(-3.0, 3.0, 0.5), np.arange(-3.0, 3.0, 0.5))
    X,Y=np.meshgrid(np.arange(min(x_lpdall), max(x_lpdall), 1), np.arange(min(y_lpdall), max(y_lpdall), 1))
    
    # evaluate it on grid
    Z = C[0]*X + C[1]*Y + C[2]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
    ax.scatter(x_lpdall, y_lpdall, z_lpdall, c='r', s=50)
    ax.scatter(x_lpdall+vec_lpdall[:,0],
               y_lpdall+vec_lpdall[:,1],
               z_lpdall+vec_lpdall[:,2],
               c='g', s=50)
    
    ax.scatter(v_normal[0]+np.mean(x_lpdall), v_normal[1]+np.mean(y_lpdall), v_normal[2]+np.mean(z_lpdall), c='b', s=50)
    ax.scatter(0+np.mean(x_lpdall), 0+np.mean(y_lpdall), 0+np.mean(z_lpdall), c='b', s=50)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_zlabel('Z')
    ax.axis('equal')
    ax.axis('tight')
    plt.show()
    '''
    
        
        
        #print(x_lpd1_in_cell, y_lpd1_in_cell, z_lpd1_in_cell)
        #print(np.shape(lpd1_cell_idx[:] == idx), (lpd1_cell_idx[:] == idx).reshape((len(lpd1_cell_idx[:]))))
    '''
        #define the cholesterol 
        # ID center of geometry coordinates for cholesterol on indicated bilayer side
    lpd3_coords = np.zeros((len(lpd3i.resnums),3))
    lpd3_res = np.zeros((len(lpd3i.resnums),1))
    lpd3_u =np.zeros((len(lpd3i.resnums),1)) 
    for i in np.arange(len(lpd3i.resnums)):
        resnum = lpd3i.resnums[i]
        head_CHL = u.select_atoms('resnum %i and (name C1)'%resnum).center_of_geometry() 
        tail_CHL = u.select_atoms('resnum %i and (name C65)'%resnum).center_of_geometry()
        vect_CHL = head_CHL- tail_CHL
        theta_CHL = np.arccos(np.dot(vect_CHL, vref)/(norm(vect_CHL)*norm(vref)))
        u_CHL= 1.5*((pow(np.cos(theta_CHL), 2)))-0.5
        group = u.select_atoms('resnum %i'%resnum)
        group_cog = group.center_of_geometry()
        lpd3_coords[i] = group_cog
        lpd3_res[i] = resnum
        lpd3_u[i] = u_CHL
          
            
            
    # ID coordinates for lipids on indicated bilayer side, renaming variables
    lpd1_coordsA = np.zeros((len(lpd1i.resnums),3))
    #lpd1_coordsB = np.zeros((len(lpd1i.resnums),3))
    lpd1_res = np.zeros((len(lpd1i.resnums),1))
    lpd1_u = np.zeros((len(lpd1i.resnums),1))
    for i in np.arange(len(lpd1i.resnums)):
        resnum_A = lpd1i.resnums[i]
        #resnum_B = lpd1i.resnums[i]
    ###test the lipids-end-tails
        head_DPPC_A = u.select_atoms('resnum %i and (name C22)'%resnum_A).center_of_geometry() 
        tail_DPPC_A = u.select_atoms('resnum %i and (name 6C21)'%resnum_A).center_of_geometry()
        vect_DPPC = head_DPPC_A - tail_DPPC_A
        theta_DPPC = np.arccos(np.dot(vect_DPPC, vref)/(norm(vect_DPPC)*norm(vref)))
        print(theta_DPPC)
        u_DPPC= ((3/2)*(pow(np.cos(theta_DPPC), 2))) - 0.5
        #print(u_DPPC)
        lpd1_u[i] = u_DPPC
        group_lpd1_chainA = u.select_atoms('resnum %i'%resnum_A)
        group_cog_lpd1A = group_lpd1_chainA.center_of_geometry()
        lpd1_coordsA[i] = group_cog_lpd1A
        lpd1_res[i] = resnum_A
        
        
    lpd2_coordsA = np.zeros((len(lpd2i.resnums),3))
    lpd2_res = np.zeros((len(lpd2i.resnums),1))
    lpd2_u = np.zeros((len(lpd2i.resnums),1))
    for i in np.arange(len(lpd2i.resnums)):
        resnumB_A = lpd2i.resnums[i]
        head_DAPC_A = u.select_atoms('resnum %i and (name C22)'%resnumB_A).center_of_geometry() 
        tail_DAPC_A = u.select_atoms('resnum %i and (name 0C22)'%resnumB_A).center_of_geometry()
        vect_DAPC = head_DAPC_A - tail_DAPC_A
        theta_DAPC = np.arccos(np.dot(vect_DAPC, vref)/(norm(vect_DAPC)*norm(vref)))
        u_DAPC= ((3/2)*(pow(np.cos(theta_DAPC), 2))) - 0.5
        lpd2_u[i] = u_DAPC
        group_lpd2_chainA = u.select_atoms('resnum %i'%resnumB_A)
        group_cog_lpd2A = group_lpd2_chainA.center_of_geometry()
        lpd2_coordsA[i] = group_cog_lpd2A
        lpd2_res[i] = resnumB_A
     
    lpd_coords = np.vstack((lpd1_coordsA,lpd1_coordsA,lpd2_coordsA,lpd2_coordsA,lpd3_coords)) #append
    lpd_resids = np.vstack((lpd1_res,lpd1_res, lpd2_res, lpd2_res, lpd3_res)) 
    lpd_us =np.vstack((lpd1_u,lpd1_u, lpd2_u,lpd2_u, lpd3_u))
    lpd_coords = lpd_coords.astype('float32')
    lpd_resids = lpd_resids.astype('float32')
    lpd_us = lpd_us.astype('float32')
    return lpd_coords,lpd_resids,lpd_us, box
    '''

#coordinates, residues, directors, box = get_side_coordinates_and_box(u)


for ts in u.trajectory[1499:1500:2]:
    coordinates, residues, directors, box = get_side_coordinates_and_box(u)
    get_side_coordinates_and_box(u)
    
    if side == 'up':
        np.save('directors_upper_tail.npy', directors)
        np.save('coordinates_upper_tail.npy', coordinates)
        np.save('residues_upper_tail.npy', residues)
    elif side == 'down':
        np.save('directors_lower_tail.npy', directors)
        np.save('coordinates_lower_tail.npy', coordinates)
        np.save('residues_lower_tail.npy', residues)

