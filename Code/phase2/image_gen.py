import bpy
from math import sin, cos, pi
import os
import numpy as np

homedir = os.path.dirname(bpy.data.filepath)
cam = bpy.context.scene.camera
cam.rotation_mode = 'XYZ'
scene = bpy.context.scene
scene.render.resolution_x = 224  # width in pixels
scene.render.resolution_y = 224  # height in pixels

def gen_images(cam, sourcepath, destpath):
    traj = np.load(sourcepath)
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
        
        gen_images(cam, homedir+f'\\train\\imu{i}.npy', f'\\train\\imgs{i}\\')
    
