import pydicom
import os
import numpy as np
import bpy
import pyopenvdb as openvdb

#import sys
#def install(package):
#    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
#install('open3d')


def create_custom_shader():
    # Get the active material or create a new one if it doesn't exist
    material = bpy.context.active_object.active_material
    if not material:
        material = bpy.data.materials.new(name="US_Volume_Material")
        bpy.context.active_object.active_material = material
        
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    
    # Clear existing nodes
    nodes.clear()
    
    # Create nodes
    node_volume_info = nodes.new(type='ShaderNodeVolumeInfo')
    node_color_ramp = nodes.new(type='ShaderNodeValToRGB')
    node_principled_volume = nodes.new(type='ShaderNodeVolumePrincipled')
    node_material_output = nodes.new(type='ShaderNodeOutputMaterial')
    
    # Set up Color Ramp node
    node_color_ramp.color_ramp.elements[0].position = 0
    node_color_ramp.color_ramp.elements[1].position = 0.241
    
    # Set up Principled Volume node
    node_principled_volume.inputs['Density'].default_value = 0
    node_principled_volume.inputs['Anisotropy'].default_value = 0
    node_principled_volume.inputs['Blackbody Intensity'].default_value = 0
    node_principled_volume.inputs['Temperature'].default_value = 1000
    
    # Connect nodes
    links.new(node_volume_info.outputs['Density'], node_color_ramp.inputs['Fac'])
    links.new(node_color_ramp.outputs['Color'], node_principled_volume.inputs['Emission Strength'])
    links.new(node_color_ramp.outputs['Color'], node_principled_volume.inputs['Emission Color'])
    links.new(node_principled_volume.outputs['Volume'], node_material_output.inputs['Volume'])

    
    # Position nodes in the shader editor
    node_volume_info.location = (-600, 0)
    node_color_ramp.location = (-400, 200)
    node_principled_volume.location = (-200, 0)
    node_material_output.location = (200, 0)

def scene_cleanup():
    """
    Cleans up memory, and deletes all objects in the scene
    """
    # Clean up unused data blocks
    bpy.ops.outliner.orphans_purge(do_recursive=True)

    # Deselect all objects
    bpy.ops.object.select_all(action='DESELECT')

    # Select objects to delete (excluding lights and cameras)
    for obj in bpy.data.objects:
        if obj.type not in ['LIGHT', 'CAMERA']:
            obj.select_set(True)
            
    bpy.ops.object.delete()


def generate_point_cloud(grid, threshold, transform, spacing=1):
    voxel_dimensions = grid.evalActiveVoxelDim()
    volume = np.zeros(voxel_dimensions)
    grid.copyToArray(volume)
    points = []
    for x in range(0, voxel_dimensions[0], spacing):
        for y in range(0, voxel_dimensions[1], spacing):
            for z in range(0, voxel_dimensions[2], spacing):
                # Get voxel value
                index = x + y*voxel_dimensions[0] + z*voxel_dimensions[0]*voxel_dimensions[1]
                voxel_value = volume[x][y][z]
                
                if voxel_value > threshold:
                    # Convert voxel coordinates to local 0-1 range
                    local_coord = np.array([x / voxel_dimensions[0], 
                                            y / voxel_dimensions[1], 
                                            z / voxel_dimensions[2], 
                                            1])  # Homogeneous coordinates
                    # Convert to global coordinates
                    global_coord = transform @ local_coord
                    points.append(global_coord[:3])  # Exclude the homogeneous component
                

    return points


def main():
    
    scene_cleanup()
    
    # TODO: change this to your local dir
    directory = r'/Users/ahmedkadri/Documents/Lectures/RCI_Prakitkum/ultra_splatting/spine_phantom/dicomdir'
    if not os.path.exists(directory):
        print(f"{directory} doesn't exist")
        exit
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if "patient" in file:
                file_paths.append(os.path.join(root,file))
    image_files = sorted(file_paths, key=lambda x: pydicom.read_file(x).InstanceNumber)

    # creates an empty array to hold US image slices
    image_planes = []
    # Define the slice thickness
    slice_spacing = None
    spacing = None
    positions = []
    orientations = []

    for image_file in image_files:
        # Load the DICOM dataset
        ds = pydicom.read_file(image_file)
        
        # Extract Per Slice Position & Orientation
        RTdata = ds[0x5200, 0x9230] # Per-frame Functional Groups Sequence
        for i in range(len(RTdata.value)):
            T = RTdata[i][0x20,0x9113][0][0x20,0x32].value # Plane position
            R = RTdata[i][0x20,0x9116][0][0x20,0x37].value # Plane orientation
            positions.append(T)
            orientations.append(R)
        
        # for i in range(len(RT)):
        #     print(f"Patient {i}: ")
        #     print(RT[i])
        # print(len(RT))

        # Extract the pixel data and metadata
        volume = ds.pixel_array
        print(volume.shape)
        
        # Pixel spacing in X,Y direction
        spacing = ds[0x5200, 0x9229][0][0x28,0x9110][0][0x28,0x30].value # Pixel Spacing
        print(spacing)
        
        # Origin coordinates (consider the first frame)
        origin = np.asarray(positions[0])
        
        # Gets the slice thickness (no information given therefore chosen as 1)
        slice_spacing = 1
        
    # Gets total number of slices
    num_slices = len(positions)

    # # Converts slice spacing from string to float
    # slice_spacing = np.asarray(slice_spacing)

    # # Converts list to numpy array
    # image_planes = np.asarray(image_planes)

    # # Normalize the image volume in range 0,1
    volume = volume / np.max(volume)

    grid = openvdb.DoubleGrid()

    # Copies image volume from numpy to VDB grid
    grid.copyFromArray(volume.astype(float))

    # Scales the grid to slice thickness and pixel size using modified identity transformation matrix
    grid.transform = openvdb.createLinearTransform([[slice_spacing/100,0,0,0], [0,spacing[0]/100,0,0], [0,0,spacing[1]/100,0], [0,0,0,1]])

    # Sets the grid class to FOG_VOLUME
    grid.gridClass = openvdb.GridClass.FOG_VOLUME

    # Blender needs grid name to be "Density"
    grid.name = "density"
    
    # Writes volume to a vdb file
    # Set output directory
    #TODO: change this line to your local dir
    output_dir = '/Users/ahmedkadri/Projekts/ultra-splatting/scripts/gaussians_initialization'
    openvdb.write(os.path.join(output_dir,'USVolume.vdb'), grid)

    # Add the volume to the scene
    bpy.ops.object.volume_import(filepath=os.path.join(output_dir,'USVolume.vdb'), files=[])

    # Set the volume's origin to match the DICOM image position
    print(origin)
    bpy.context.object.location = origin / 100
    
    create_custom_shader()
    world_transform = bpy.context.object.matrix_world
    world_transform = np.array(world_transform).reshape(4,4)
    points = grid.convertToPolygons(isovalue=80)
    print(f'iso = {80}, vertices = {points[0]}')
    print(f'active voxels = {grid.evalActiveVoxelDim()}')
    print(f'min/max = {grid.evalMinMax()}')
    points = generate_point_cloud(grid=grid, threshold=0.5, transform=world_transform, spacing=1)
    points = np.array(points)
    print(points.shape)
    np.save(os.path.join(output_dir,'volume_pcl.npy'),points)
    
if "__main__" == __name__:
    main()