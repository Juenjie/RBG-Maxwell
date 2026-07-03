#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os,sys
import numpy as np
def create_folder(folder_name):
    '''create an empty folder with given name folder_name
    
    Params
    ======
    floder_name: str, name of the folder
       
    code: folder = os.getcwd()+'/'+folder_name+'/'   
    '''
    
    folder = os.getcwd()+'/'+folder_name+'/'
    
    if not os.path.exists(folder):
        os.makedirs(folder)
            
    # clean temp file
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        if os.path.isfile(file_path):
            os.unlink(file_path)


def save_file(file_name,file):
    
    '''
    Give the file_name and file, save it under folder ZBEsolver_data

    Params
    ======
    file_name: str, eg. 'BEsolverData/state/0'
    '''    
    file.astype('float64').tofile(os.getcwd()+file_name+'.dat')



