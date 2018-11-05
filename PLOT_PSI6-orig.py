
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt


# In[5]:


#get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


#read the vectors


# In[49]:


psi6_value_upper= np.load("psi6s_upper_tail.npy")


# In[50]:


psi6_U=np.absolute(psi6_value_upper)


# In[51]:


psi6_U[0].shape


# In[52]:


coords_value_upper=np.load("coord_upper_tail.npy")


# In[53]:


coords_value_upper[0].shape


# In[54]:


resid_value_upper=np.load("resids_upper_tail.npy")


# In[55]:


for i in range(len(psi6_U)):
    resid_U=(resid_value_upper[i]).T
    sg=np.where(np.logical_and(resid_U>=102, resid_U<=354))[1]
    tg=np.where(np.logical_and(resid_U>=355, resid_U<=506))[1]
    cg=np.where(np.logical_or(resid_U>=506, resid_U<=102))[1]
    pos=coords_value_upper[i]
    plt.figure(figsize=(15,5))
    plt.title('Frame '+str(i),size=20)
    plt.scatter(pos[:,0],pos[:,1],vmin=0.0,vmax=1.0,
            c=(psi6_U[i]),s=50,cmap=plt.cm.GnBu)
    cb=plt.colorbar()

    plt.scatter(pos[sg,0],pos[sg,1],facecolors='none', edgecolors='#CC0000', s=130, lw=1.5)
    plt.scatter(pos[tg,0],pos[tg,1],facecolors='none', edgecolors='black', s=130, lw=1.5)
    plt.scatter(pos[cg,0],pos[cg,1],facecolors='none', edgecolors='#F5C816', s=130, lw=1.5)
    plt.xlim(np.nanmin(pos[:,0])-1,np.nanmax(pos[:,0])+1)
    plt.ylim(np.nanmin(pos[:,1])-1,np.nanmax(pos[:,1])+1)
    cb.set_label(label='$|\psi_6|$',size=20)
    cb.ax.tick_params(labelsize=16)
    plt.tick_params(axis='x', pad=8)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.savefig("frame" + str(i) + '.png', dpi=300)
    

