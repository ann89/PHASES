
# coding: utf-8

# In[32]:


import MDAnalysis
import MDAnalysis.lib.NeighborSearch as NS
import numpy as np
import sys
import multiprocessing as mp
from time import time
from numpy.linalg import norm


# In[16]:


top  = 'membrane-start.pdb'
traj = '150_ns_membrane_whole-100ps.xtc'
side = "up" #sys.argv[1] # "up" for upper leaflet "down" for lower leaflet
skip = 14000


# In[17]:


u = MDAnalysis.Universe(top,traj)
n_frames = u.trajectory.n_frames
frames = np.arange(n_frames)[::skip]


# In[26]:


# norm vector along the z-axis
vref = np.array([0.0,0.0,1.0])


# In[ ]:


#def u_CHL(u):
    #   """Calculate the director parameter for Cholesterol"""
    


# In[44]:


def get_side_coordinates_and_box(frame):
    """Assign lipids to leaflets, retrieve their coordinates, resIDs and the director order parameter u=3/2cosˆ2(theta)-1/2"""
    frame = 1499
    #u = MDAnalysis.Universe(top,traj)
    u.trajectory[1499:1500:2]

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
    lpd1_coordsB = np.zeros((len(lpd1i.resnums),3))
    lpd1_res = np.zeros((len(lpd1i.resnums),1))
    lpd1_u = np.zeros((len(lpd1i.resnums),1))
    for i in np.arange(len(lpd1i.resnums)):
        resnum_A = lpd1i.resnums[i]
        resnum_B = lpd1i.resnums[i]
    ###test the lipids-end-tails
        head_DPPC_A = u.select_atoms('resnum %i and (name C22)'%resnum_A).center_of_geometry() 
        tail_DPPC_A = u.select_atoms('resnum %i and (name 6C21)'%resnum_A).center_of_geometry()
        vect_DPPC_A = head_DPPC_A - tail_DPPC_A
        theta_DPPC_A = np.arccos(np.dot(vect_DPPC_A, vref)/(norm(vect_DPPC_A)*norm(vref)))
        
        head_DPPC_B = u.select_atoms('resnum %i and (name C32)'%resnum_B).center_of_geometry() 
        tail_DPPC_B = u.select_atoms('resnum %i and (name 6C31)'%resnum_B).center_of_geometry()
        vect_DPPC_B = head_DPPC_B - tail_DPPC_B
        theta_DPPC_B = np.arccos(np.dot(vect_DPPC_B, vref)/(norm(vect_DPPC_B)*norm(vref))) 
             
        theta_DPPC = (theta_DPPC_A + theta_DPPC_B)/2
        u_DPPC= ((3/2)*(pow(np.cos(theta_DPPC), 2))) - 0.5
        #print(u_DPPC)
        lpd1_u[i] = u_DPPC
        group_lpd1_chainA = u.select_atoms('resnum %i and (name C22 or name 6C21)'%resnum_A)
        group_lpd1_chainB = u.select_atoms('resnum %i and (name C32 or name 6C21)'%resnum_B)
        group_cog_lpd1A = group_lpd1_chainA.center_of_geometry()
        group_cog_lpd1B = group_lpd1_chainB.center_of_geometry()
        lpd1_coordsA[i] = group_cog_lpd1A
        lpd1_coordsB[i] = group_cog_lpd1B
        lpd1_res[i] = resnum_A
        
        
    lpd2_coordsA = np.zeros((len(lpd2i.resnums),3))
    lpd2_coordsB = np.zeros((len(lpd2i.resnums),3))
    lpd2_res = np.zeros((len(lpd2i.resnums),1))
    lpd2_u = np.zeros((len(lpd2i.resnums),1))
    for i in np.arange(len(lpd2i.resnums)):
        resnumB_A = lpd2i.resnums[i]
        resnumB_B = lpd2i.resnums[i] 
        
        head_DAPC_A = u.select_atoms('resnum %i and (name C22)'%resnumB_A).center_of_geometry() 
        tail_DAPC_A = u.select_atoms('resnum %i and (name 0C22)'%resnumB_A).center_of_geometry()
        vect_DAPC_A = head_DAPC_A - tail_DAPC_A
        theta_DAPC_A = np.arccos(np.dot(vect_DAPC_A, vref)/(norm(vect_DAPC_A)*norm(vref)))
        
        head_DAPC_B = u.select_atoms('resnum %i and (name C32)'%resnumB_B).center_of_geometry() 
        tail_DAPC_B = u.select_atoms('resnum %i and (name 0C32)'%resnumB_B).center_of_geometry()
        vect_DAPC_B = head_DAPC_B - tail_DAPC_B
        theta_DAPC_B = np.arccos(np.dot(vect_DAPC_B, vref)/(norm(vect_DAPC_B)*norm(vref)))
             
        theta_DAPC = (theta_DAPC_A + theta_DAPC_B)/2     

        u_DAPC= ((3/2)*(pow(np.cos(theta_DAPC), 2))) - 0.5
        lpd2_u[i] = u_DAPC
        group_lpd2_chainA = u.select_atoms('resnum %i and (name C22 or name 0C22)'%resnumB_A)
        group_lpd2_chainB = u.select_atoms('resnum %i and (name C32 or name 0C32)'%resnumB_B)
        group_cog_lpd2A = group_lpd2_chainA.center_of_geometry()
        lpd2_coordsA[i] = group_cog_lpd2A
        group_cog_lpd2B = group_lpd2_chainB.center_of_geometry()
        lpd2_coordsB[i] = group_cog_lpd2B
        lpd2_res[i] = resnumB_A
     
    lpd_coords = np.vstack((lpd1_coordsA,lpd1_coordsB,lpd2_coordsA,lpd2_coordsB,lpd3_coords)) #append
    lpd_resids = np.vstack((lpd1_res,lpd1_res, lpd2_res, lpd2_res, lpd3_res)) 
    lpd_us =np.vstack((lpd1_u,lpd1_u, lpd2_u,lpd2_u, lpd3_u))
    lpd_coords = lpd_coords.astype('float32')
    lpd_resids = lpd_resids.astype('float32')
    lpd_us = lpd_us.astype('float32')
    return lpd_coords,lpd_resids,lpd_us, box




coordinates, residues, directors, box = get_side_coordinates_and_box(frames)
if side == 'up':
    np.save('directors_upper_tail-T.npy', directors)
    np.save('coordinates_upper_tail-T.npy', coordinates)
    np.save('residues_upper_tail-T.npy', residues)
elif side == 'down':
    np.save('directors_lower_tail.npy', directors)
    np.save('coordinates_lower_tail.npy', coordinates)
    np.save('residues_lower_tail.npy', residues)
  