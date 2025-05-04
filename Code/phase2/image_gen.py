import bpy
import numpy as np
from math import sin, cos, pi
import time
import os
from trajectory_gen import generate_num_data


cam = bpy.context.scene.camera
cam.rotation_mode = 'XYZ'
homedir = os.path.dirname(bpy.data.filepath)
scene = bpy.context.scene
scene.render.resolution_x = 224  # width in pixels
scene.render.resolution_y = 224  # height in pixels

def gen_images(cam, traj, destpath):
    for i in range(0, traj.shape[0], 10):
        point = traj[i]
        bpy.context.scene.render.filepath = homedir+destpath+f'{i//10}.png'
        
        loc = point[1:4]
        orient = point[4:]
    
        cam.rotation_euler = orient
        cam.location = loc
        bpy.ops.render.render(write_still=True)
        
        

if __name__ == '__main__':
    ntraj = 1

    for i in range(ntraj):
        pose = generate_num_data(i)
        gen_images(cam, pose, f'\\imgs{i}\\')
    