
# coding: utf-8

# In[1]:


import MDAnalysis
import MDAnalysis.lib.NeighborSearch as NS
import numpy as np
import sys
import multiprocessing as mp
from time import time


# In[62]:


# input
nprocs = 1
top  = 'membrane-start.pdb'
traj = '150_ns_membrane_whole-100ps.xtc'
side = "up" #sys.argv[1] # "up" for upper leaflet "down" for lower leaflet
skip = 1400


# In[63]:


u = MDAnalysis.Universe(top,traj)
n_frames = u.trajectory.n_frames
frames = np.arange(n_frames)[::skip]


# In[64]:


# set reference vector
vref = np.array([[1,0,0],[0,0,0]])


# In[79]:


def get_side_coordinates_and_box(frame):
    #u = MDAnalysis.Universe(top,traj)
    u.trajectory[frame]

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
        ##print(lpd1i)
        ##print(lpd3i)
    elif side == 'down':
        lpd1i = lpd1_atoms[int((num_lpd1)/2):]
        lpd2i = lpd2_atoms[int((num_lpd2)/2):]
        lipids = lpd1i + lpd2i
        ns_lipids = NS.AtomNeighborSearch(lpd3_atoms)
        lpd3i = ns_lipids.search(lipids,15.0)

    
    # ID center of geometry coordinates for cholesterol on indicated bilayer side
    lpd3_coords = np.zeros((len(lpd3i.resnums),3))
    lpd3_res = np.zeros((len(lpd3i.resnums),1)) 
    for i in np.arange(len(lpd3i.resnums)):
        resnum = lpd3i.resnums[i]
        #group = u.select_atoms('resnum %i and (name C67)'%resnum)
        group = u.select_atoms('resnum %i and (name C56 or name C50 or name C24 or name C65 or name C62 or name C67 or name C59 or name C9 or name C11 or name C14 or name C16 or name C18 or name C21 or name C24 or name C26 or name 27 or name C31 or name C34 or name C37 or name C39 or name C40 or name C44)'%resnum)
        group_cog = group.center_of_geometry()
        lpd3_coords[i] = group_cog
        lpd3_res[i] = resnum
    # ID coordinates for lipids on indicated bilayer side, renaming variables
    lpd1_coordsA = np.zeros((len(lpd1i.resnums),3))
    lpd1_coordsB = np.zeros((len(lpd1i.resnums),3))
    lpd1_res = np.zeros((len(lpd1i.resnums),1))
    for i in np.arange(len(lpd1i.resnums)):
        resnum_A = lpd1i.resnums[i]
        resnum_B = lpd1i.resnums[i]
#test the lipids-end-tails
        group_lpd1_chainA = u.select_atoms('resnum %s and (name 6C21 or name C22)'%resnum_A)
        group_lpd1_chainB = u.select_atoms('resnum %s and (name 6C31 or name C32)'%resnum_A)
        #group_lpd1_chainA = u.select_atoms('resnum %s and (name C26 or name C27 or name C28 or name C29)'%resnum_A)
        #group_lpd1_chainB = u.select_atoms('resnum %s and (name C36 or name C37 or name C38 or name C39)'%resnum_B)
        group_cog_lpd1A = group_lpd1_chainA.center_of_geometry()
        group_cog_lpd1B = group_lpd1_chainB.center_of_geometry()
        lpd1_coordsA[i] = group_cog_lpd1A
        lpd1_coordsB[i] = group_cog_lpd1B
        lpd1_res[i] = resnum_A

    
    
    lpd2_coordsA = np.zeros((len(lpd2i.resnums),3))
    lpd2_coordsB = np.zeros((len(lpd2i.resnums),3))
    lpd2_res = np.zeros((len(lpd2i.resnums),1))
    for i in np.arange(len(lpd2i.resnums)):
        resnumB_A = lpd2i.resnums[i]
        resnumB_B = lpd2i.resnums[i]
        group_lpd2_chainA = u.select_atoms('resnum %s and (name 0C22 or name C22)'%resnumB_A)
        group_lpd2_chainB = u.select_atoms('resnum %s and (name 0C32 or name C32)'%resnumB_B)
        
        #group_lpd2_chainA = u.select_atoms('resnum %s and (name C26 or name C27 or name C28 or name C29)'%resnumB_A)
        #group_lpd2_chainB = u.select_atoms('resnum %s and (name C36 or name C37 or name C38 or name C39)'%resnumB_B)
        group_cog_lpd2A = group_lpd2_chainA.center_of_geometry()
        group_cog_lpd2B = group_lpd2_chainB.center_of_geometry()
        lpd2_coordsA[i] = group_cog_lpd2A
        lpd2_coordsB[i] = group_cog_lpd2B 
        lpd2_res[i] = resnumB_A



    #lpd1_atoms = u.select_atoms('resname %s and (name C26 or name C36 )'%lipid1)
    #lpd2_atoms = u.select_atoms('resname %s and (name C24 or name C34)'%lipid2)
    #num_lpd1 = lpd1_atoms.n_atoms
    #num_lpd2 = lpd2_atoms.n_atoms
    
    # select lipid tail atoms beloging to the selected bilayer side
   # if side == 'up':
      #  lpd1i = lpd1_atoms[:int((num_lpd1)/2)]
        #lpd2i = lpd2_atoms[:int((num_lpd2)/2)]
    
   # elif side == 'down':
       # lpd1i = lpd1_atoms[int((num_lpd1)/2):]
    #    lpd2i = lpd2_atoms[int((num_lpd2)/2):]
    
    # assign lpd1 and lpd2 coordinates, completing the assignment of all coordinates from which psi6 will be computed
    ###lpd1_coords = lpd1i.positions
    #lpd2_coords = lpd2i.positions
    #lpd_coords = np.vstack((lpd1_coords,lpd2_coords))
    lpd_coords = np.vstack((lpd1_coordsA,lpd1_coordsB,lpd2_coordsA,lpd2_coordsB,lpd3_coords)) #append
    lpd_resids = np.vstack((lpd1_res,lpd1_res,lpd2_res,lpd2_res,lpd3_res)) 
    ##lpd_coords = np.vstack((lpd1_coords,lpd2_coords,lpd3_coords)) #append
    lpd_coords = lpd_coords.astype('float32')
    lpd_resids = lpd_resids.astype('float32')
    return lpd_coords,lpd_resids, box


# In[80]:


def standard_fit(X):
    # Find the average of points (centroid) along the columns
    C = np.average(X, axis=0)
    # Create CX vector (centroid to point) matrix
    CX = X - C
    # Singular value decomposition
    U, S, V = np.linalg.svd(CX)
    # The last row of V matrix indicate the eigenvectors of
    # smallest eigenvalues (singular values).
    N = V[-1]
    return C, N


# In[81]:


# The projection of these points onto the best-fit plane
def projection(x, C, N):
    rows, cols = x.shape
    NN = np.tile(N, (rows, 1))
    D = np.dot(x-C, N)
    DD = np.tile(D, (cols, 1)).T
    return x - DD * NN


# In[82]:


def dist(vec):
    distance = np.sqrt(np.power(vec[0],2) + np.power(vec[1],2) + np.power(vec[2],2))
    return distance


# In[83]:


def angles_normvec_psi6(coords, atom, box):
    distarr = MDAnalysis.lib.distances.distance_array(coords,coords,box=box)
    
    nn_inds = np.argsort(distarr[atom])[0:7] # the first index is the reference atom
    centered_coords = np.zeros((7,3))
    center_coord = coords[atom]
    for i in np.arange(7):
        centered_coords[i] = coords[nn_inds[i]] - center_coord

    C, N = standard_fit(centered_coords)

    projected_coords = projection(centered_coords,C,N)
    projected_vref = projection(vref,C,N)[0]

    centered_coords = projected_coords - projected_coords[0]
    centered_vref = projected_vref - projected_coords[0]
    centered_N = N - projected_coords[0]

    angles = np.zeros(6)
    for neighbor in np.arange(1,7):
        # compute the angle against the reference vector
        norm = dist(centered_vref)*dist(centered_coords[neighbor])
        angle = np.arccos(np.dot(centered_vref,centered_coords[neighbor])/norm)
        if np.isnan(angle) == True:
            angle = 0.0
        # check whether angle belongs to 1st and 3rd or 2nd and 4th circle quadrants using 
        # a little trick with the normal vector
        if np.dot(centered_N,np.cross(centered_vref,centered_coords[neighbor])) < 0.0:
            angle = (np.pi*2) - angle
        angles[neighbor-1] = angle

    psi6 = np.mean( np.cos(angles*6) + (1j*np.sin(angles*6)))
    
    return psi6, angles


# In[84]:


def get_psi6(frame):
    start = time()
    print('Finding psi6, normvec in frame %i of %i'%(frame, n_frames))
    coords, resids, box = get_side_coordinates_and_box(frame)
    print('Neighbors search finished after', time() - start, 'seconds')
    n_atoms = coords.shape[0]
    psi6s = np.zeros(n_atoms,dtype=complex)
    angles = np.zeros((n_atoms,6))
    for atom in np.arange(n_atoms):
        psi6s[atom], angles[atom] = angles_normvec_psi6(coords, atom, box)
    print('Finished after', time() - start, 'seconds')
    return psi6s, angles, coords, resids


# In[85]:


pool = mp.Pool(processes=nprocs)
print ('Initiating multiprocessing with %i processors'%nprocs)
results = pool.map(get_psi6, frames)

atom_angles = []
atom_psi6s  = []
atom_coord  = []
atom_resids = []
for i in range(len(results)):
    atom_psi6s.append(results[i][0])
    atom_angles.append(results[i][1])
    atom_coord.append(results[i][2])
    atom_resids.append(results[i][3])
# write out the complex vector computed for psi6 and also
# write out both the angles to each neighbor of each particle
if side == 'up':
    np.save('psi6s_upper_tail.npy', atom_psi6s)
    np.save('angles_upper_tail.npy', atom_angles)
    np.save('coord_upper_tail.npy', atom_coord)
    np.save('resids_upper_tail.npy', atom_resids)
elif side == 'down':
    np.save('psi6s_lower_tail.npy', atom_psi6s)
    np.save('angles_lower_tail.npy', atom_angles)
    np.save('coord_lower_tail.npy', atom_coord)
    np.save('resids_test-lower.npy', atom_resids)


# In[26]:


#PLOT STUFF


# In[14]:


psi6_value= np.load("psi6s_upper_tail.npy")


# In[15]:


positions_value= np.load("coord_upper_tail.npy")


# In[16]:


resids_value =np.load("resids_upper_tail.npy")


# In[89]:


res= resids_value[1]


# In[90]:


s=np.array(res).T


# In[91]:


s.shape


# In[92]:


b=np.absolute(psi6_value)

b.shape


# In[99]:


print(np.mean(b[0]))


# In[103]:


frame1= positions_value[0]
print(frame1)


# In[104]:


frame1.shape


# In[102]:


y = np.stack(frame1)
print(y)


# In[81]:


y.shape


# In[82]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[83]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[84]:


print(s)


# In[85]:


sg=np.where(np.logical_and(s>=102, s<=354))[1]
tg=np.where(np.logical_and(s>=355, s<=506))[1]
cg=np.where(np.logical_or(s>=506, s<=102))[1]
print(cg)


# In[87]:


plt.figure(figsize=(15,5))
plt.title('$\psi_6$',size=20)
plt.scatter(y[:,0],y[:,1],vmin=0.0,vmax=1.0,
            c=(b[0]),s=50,cmap=plt.cm.GnBu)
cb=plt.colorbar()

plt.scatter(y[sg,0],y[sg,1],facecolors='none', edgecolors='#CC0000', s=130, lw=1.5)
plt.scatter(y[tg,0],y[tg,1],facecolors='none', edgecolors='black', s=130, lw=1.5)
plt.scatter(y[cg,0],y[cg,1],facecolors='none', edgecolors='#F5C816', s=130, lw=1.5)
plt.xlim(np.nanmin(y[:,0])-1,np.nanmax(y[:,0])+1)
plt.ylim(np.nanmin(y[:,1])-1,np.nanmax(y[:,1])+1)
cb.set_label(label='$|\psi_6|$',size=20)
cb.ax.tick_params(labelsize=16)
plt.tick_params(axis='x', pad=8)
plt.tick_params(axis='both', which='major', labelsize=16)


# In[77]:


np.where(np.logical_and(s>=102, s<=354))[1]


# In[313]:


s[0,2]


# In[32]:


from scipy import interpolate
from scipy.interpolate import griddata
#from matplotlib.mlab import griddata


# In[33]:


x13=np.arange(0,302,2)
y13=np.arange(0,102,2)
zi13=griddata((y[:,0], y[:,1]), b[2],(x13[None,:] ,y13[:,None]), method='linear')


# In[34]:


v13 = np.linspace(0, 1, 11, endpoint=True)
plt.figure(figsize=(15,5))
plt.contourf(x13, y13, zi13,v13,cmap=plt.cm.viridis)
cb=plt.colorbar()
cb.set_label(label='$|\psi_6|$',size=20)
cb.ax.tick_params(labelsize=16)
plt.tick_params(axis='x', pad=8)
plt.tick_params(axis='both', which='major', labelsize=16)


# In[ ]:


#Make data interpolation for the plot


# In[ ]:


#make the grid and evaluat the function


# In[ ]:


import numpy as np

def func(x, y):
    return np.sin(y * x)

xaxis = np.linspace(0, 4, 10)
yaxis = np.linspace(-1, 1, 20)
result = func(x[:,None], y[None,:])

