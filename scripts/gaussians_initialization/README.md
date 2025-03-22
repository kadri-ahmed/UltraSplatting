# Initialization using Mesh from ImFusion Suite

## 1. Apply Volume Compounding to US Sweep

You can select one or multiple sweeps and make sure to choose a suitable background intensity for better results:

![alt text](./figures/step_1_imf_volume_compounding.png "Volume Compounding ImFusion")

You should get an output similar to the following:

![alt text](./figures/step_1_imf_volume_compounding_output.png "Volume Compounding Output ImFusion")

## 2. Extract Mesh from Volume
You can extract a mesh from the volume by using Algorithms > Segmentation > Extract Mesh functionality and make sure to  choose a suitable ISO value:

![alt text](./figures/step_1_imf_volume_to_mesh.png "Extract Mesh from Volume")

## 3. Export Mesh 
![alt text](./figures/step_3_imf_export_mesh.png "Export Mesh")

## 4. Get Pointcloud from Mesh
To initialize the 3D Gaussians we can use the positions of vertices and save them as the point cloud we will use to place our initial 3D Gaussians. \

``` python mesh_to_pcl.py -i <input_mesh_path> -o <output_pcl_path>```

![alt text](./figures/step_4_mesh_to_pcl.png "Get Pointcloud from Mesh")

# Initialization using Volume Rendering in Blender

## 1. Export Sweep as a DICOM File
![alt text](./figures/step_1_export_dicomfile.png "Export Sweep as DICOM file")

## 2. Set Directory path to DICOM File
Change this line in code to your local directory: \
``` directory = r'/Users/ahmedkadri/Documents/Lectures/RCI_Prakitkum/ultra_splatting/spine_phantom/dicomdir'```

## 3A. Run script in Blender
You should get something similar to this volume: 

![alt text](./figures/step_3_blender_volrend_output.png "Blender Volume Rendering Output")

You can play around with the slider parameter to get better rendering results for your specific model:

![alt text](./figures/step_3_slider_parameter.png "Color Ramp slider")

## 4. From Volume to Pointcloud

By reading the .npy file you saved from blender, you save the 3D points as ply file to get the pointcloud, which should look something like this:
![alt text](./figures/step_4_volume_to_pcl.png "Volume to Poincloud")

## 5. Blender missing dependency
If you need to install a missing dependency into the python API of blender, add this at the top of your script:

```Python import subprocess
# Example to install pydicom package 
import sys
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install('pydicom')
```
## 6. Debugging Blender scripts
To see debug messages and console outputs of python scripts within blender, you need to launch blender from the terminal by following this manual for your operating system:
[Launching Blender from the Command Line](https://docs.blender.org/manual/en/latest/advanced/command_line/launch/index.html#command-line-launch-index) 

![alt text](./figures/step_5_blender_debug_from_terminal.png "Launch Blender from terminal")

## 7. Change the render engine from EEVEE to Cycles in Blender
Change the render engine to Cycles for better visualization quality

![alt text](./figures/step_6_set_render_engine_to_cycles.png "Render Engine Cycles")