import os
import numpy as np
import bpy
import mathutils
from functools import partial

# Function to import images as planes and apply transformations
def import_images_as_planes(image_files, folder_path):
    bpy.ops.import_image.to_plane(
        files=[{"name": f} for f in image_files],
        directory=folder_path,
    )

    # Get the list of imported planes (assuming they are selected after import)
    planes = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
        
    return planes

def adjust_scale_pose(plane, pose, image_depth_m, collection):
    plane.scale = (image_depth_m, image_depth_m, image_depth_m)
    translation = pose[:3, 3] / 1000  # Adjustment for Blender meter scale because we get pose in cms
    translation -= [0, image_depth_m / 2, 0]  # Adjust origin
    rotation = pose[:3, :3]
    rot_matrix_blender = mathutils.Matrix(rotation)
    euler_rotation = rot_matrix_blender.to_euler()
    
    plane.location = translation
    plane.rotation_euler = euler_rotation
    
    # Move plane to the new collection
    bpy.context.scene.collection.objects.unlink(plane)
    collection.objects.link(plane)

def generate_transparent_material(plane, transparency):
    if not plane.data.materials:
        raise ValueError(f"Plane has no material assigned.")
    
    mat = plane.data.materials[0]
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Create and configure the color ramp node
    color_ramp = nodes.new(type='ShaderNodeValToRGB')
    color_ramp.color_ramp.interpolation = 'CONSTANT'
    color_ramp.color_ramp.elements[0].position = 0.0
    color_ramp.color_ramp.elements[1].position = 0.01
    color_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    color_ramp.color_ramp.elements[0].color = (transparency, transparency, transparency, 1.0)
    
    # Find the Principled BSDF node
    principled_bsdf = next(node for node in nodes if isinstance(node, bpy.types.ShaderNodeBsdfPrincipled))
    if principled_bsdf is None:
        raise ValueError("No Principled BSDF node found in the material.")
    
    # Connect the color ramp to the alpha input of the Principled BSDF node
    links.new(color_ramp.outputs['Color'], principled_bsdf.inputs['Alpha'])
    
    # Find the texture node
    texture_node = next(node for node in nodes if isinstance(node, bpy.types.ShaderNodeTexImage))
    if texture_node is None:
        raise ValueError("No Image Texture node found in the material.")
    
    # Connect texture node outputs
    links.new(texture_node.outputs['Color'], color_ramp.inputs['Fac'])
    links.new(texture_node.outputs['Color'], principled_bsdf.inputs['Emission'])
    principled_bsdf.inputs['Emission Strength'].default_value = 1

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses):
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    return c2w

def recenter_poses(poses):
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)
    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses

# Main function to process the datasets
def process_datasets(folders, dataset, dataset_names, image_depth_m):
    bpy.ops.outliner.orphans_purge(do_recursive=True)

    for dataset_name in dataset_names:
        folder_path = os.path.join(folders, dataset, dataset_name, 'images')
        poses_path = os.path.join(folders, dataset, dataset_name, 'poses.npy')
        
        poses = np.load(poses_path)
        poses = recenter_poses(poses)
        image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png') or f.endswith('.jpg')])
        
        if len(image_files) != len(poses):
            raise ValueError(f"Number of images does not match number of poses for dataset {dataset_name}")
        
        new_collection = bpy.data.collections.new(dataset_name)
        bpy.context.scene.collection.children.link(new_collection)
        
        planes = import_images_as_planes(image_files, folder_path)
        
        # Use partial to create new functions with fixed arguments
        adjust_func = partial(adjust_scale_pose, image_depth_m=image_depth_m, collection=new_collection)
        transparency_func = partial(generate_transparent_material, transparency=0.0)
        
        list(map(adjust_func, planes, poses))
        list(map(transparency_func, planes))
        
        print(f"Images imported, transformed, and materials assigned successfully for dataset {dataset_name}.")

# Parameters
folders = r"C:\Users\Flora\Documents\ultra-splatting\datasets\unprocessed"
dataset = "spine_phantom"

# for the datasets each left_NUMBER corresponds with a fitting sweep right_NUMBER
# if you use datasets from different NUMBER acquisitions, the accumulated tracking error in between acquisitions makes them not fit
# the number after the first NUMBER designates if it was the first or second acquisition
dataset_names = ["left1", "left1_1", "left2", "left3", "left3_2", "right1", "right1_1", "right3", "right3_2"]

# sweep pairs that fit the best together:
dataset_names = ["left1_1", "right1"]
dataset_names = ["left3", "right3_2"]

image_depth_m = 0.13

# Process the datasets
process_datasets(folders, dataset, dataset_names, image_depth_m)
