#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os, sys, glob, abc
from matplotlib import pyplot as plt, colors
def read(filelist):
    data = []
    for f in sorted(filelist):
        x = np.load(f)
        if len(data) == 0: data = x.copy()
        else: data = np.concatenate([data, x])
    return data

# Where is the dataset stored?
dataset = ""

# Read in all of the data files
data_files = dataset + "IC86_*exp.npy"
data = read(glob.glob(data_files))

# Read in all of the MC files 
sim_files = dataset + "IC86*MC*npy"
sim = read(glob.glob(sim_files))

# Set the angular error floor to 0.2 degrees
data['angErr'][data['angErr']<np.deg2rad(1)] = np.deg2rad(1)
sim['angErr'][sim['angErr']<np.deg2rad(1)] = np.deg2rad(1)

data['ra']=np.random.uniform(0,2*np.pi,size=len(data))


# In[2]:


grl_files = dataset + "GRL/IC86_*exp.npy"
grl = read(glob.glob(grl_files))

# Show the keys available in the GRL
print("Keys available in the GoodRunList:")
print(sorted(grl.dtype.names))


# In[3]:


import mla.core
from mla.spectral import *
from mla import injection
import mla.sensitivtiy 

dec = 22.0145*np.pi/180
ra = 83.63*np.pi/180


# In[4]:


from imp import reload
import scipy.optimize
reload(mla.core)
reload( mla.sensitivtiy )

import warnings
warnings.filterwarnings("ignore")
test=mla.sensitivtiy.PS_sensitivity()
test.background_building(data,sim,save_file="")
#test.load_background(dir_name="")


# In[5]:


test.set_point_source(ra,dec,data,sim,"PowerLaw",background_time_profile = (55700,55700.01157407407
))


# In[6]:


test.set_backround(data,grl)


# In[7]:


test.set_injection(sim,background_time_profile = (55700,55700.01157407407))


# In[ ]:


test.calculate_sensitivity(list_N = np.linspace(1e-16,5e-15,15),make_plot="test.png")


# In[ ]:




