import open3d as o3d
import numpy as np
import argparse

def ply_to_pointcloud(input_file, output_file, show_pcl=False):
    # Read the PLY file
    pcd = o3d.io.read_point_cloud(input_file)
    
    # Extract the points as a numpy array
    points = np.asarray(pcd.points)
    
    # Save the points as a simple text file (you can choose other formats as needed)
    extension = ".txt"
    np.savetxt(f"{output_file}{extension}", points, delimiter=',', header='x,y,z', comments='')
    
    # Save the points as a PLY file
    extension = ".ply"
    o3d.io.write_point_cloud(f"{output_file}{extension}", pcd)
    
    print(f"Point cloud saved to {output_file}")
    
    # Optionally, visualize the point cloud
    if show_pcl:
        o3d.visualization.draw_geometries([pcd])

# Usage
if "__main__" == __name__:
    parser = argparse.ArgumentParser(
            prog="Pointcloud from Mesh",
            description="Extract 3D Points from vertices of Mesh exported using ImFusion Suite"
        )
    parser.add_argument('-i', '--input', default='/Users/ahmedkadri/Documents/Lectures/RCI_Prakitkum/ultra_splatting/spine_phantom/us_sweep_mesh.ply') # path to dataset
    parser.add_argument('-o', '--output', default='output_pointcloud')
    args = parser.parse_args()
    ply_to_pointcloud(args.input, args.output, show_pcl=True)