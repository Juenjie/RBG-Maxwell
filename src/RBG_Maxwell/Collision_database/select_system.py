import os
from .. import Collision_database

def which_system(system_name):
    with open(Collision_database.__path__[0]+'/selected_system.txt','w') as save_text:
        save_text.write(system_name)
    