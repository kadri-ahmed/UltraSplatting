import os
import random
import numpy as np
import struct
from PIL import Image
 
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
 
def write_cameras_txt(filepath, camera_id, model, width, height, fx, fy, cx, cy):
    with open(filepath, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"{camera_id} {model} {width} {height} {fx} {fy} {cx} {cy}\n")
 
def write_images_txt(filepath, poses, camera_id, image_folder, actual_image_height):
    with open(filepath, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        num_images = poses.shape[0]
        f.write(f"# Number of images: {num_images}\n")
 
        for i in range(num_images):
            pose = poses[i]
            translation = pose[:3, 3] / 1000  # Convert to meters
            rotation = pose[:3, :3]
 
            # rotate and translate to a more logical camera position
            translation_z = np.array([0, 0, actual_height/2])
            # chosen arbitrarily but impacts other factors potentially (ie projection matrix and near/farplane)
            translation_x = np.array([0, actual_height/2, 0])
            # rotate to look towards imaging plane like a camera would and not like the US transducer is directed
            rotation_x_90 = np.array([[1,0,0], [0,0,-1], [0,1,0]])
 
            translation += translation_z
            translation += translation_x
 
            rotation = rotation_x_90 @ rotation
 
            # Convert rotation matrix to quaternion
            qw, qx, qy, qz = rotmat_to_quaternion(rotation)
 
            image_name = f"image_{i+1:04d}.png"  # Assuming images are named in a specific format
            # image_path = os.path.join(image_folder, image_name)
 
            f.write(f"{i+1} {qw} {qx} {qy} {qz} {translation[0]} {translation[1]} {translation[2]} {camera_id} {image_name}\n")
            f.write("\n")  # Empty points2D as a placeholder
 
def rotmat_to_quaternion(R):
    """Converts a rotation matrix to a quaternion (qw, qx, qy, qz)."""
    qw = np.sqrt(1.0 + R[0, 0] + R[1, 1] + R[2, 2]) / 2.0
    qx = (R[2, 1] - R[1, 2]) / (4.0 * qw)
    qy = (R[0, 2] - R[2, 0]) / (4.0 * qw)
    qz = (R[1, 0] - R[0, 1]) / (4.0 * qw)
    return qw, qx, qy, qz
 
def write_colmap_ini(filepath, project_dir, database_path=""):
    with open(filepath, 'w') as f:
        f.write("# Automatically generated COLMAP project file\n")
        f.write("[General]\n")
        # f.write(f"database_path={database_path}\n")
        f.write(f"image_path={project_dir}\n")
        f.write(f"export_path={project_dir}\n")
        f.write("\n")
        f.write("[Options]\n")
        f.write("use_vocabulary_tree=1\n")
 
def process_and_save_images(images_npy, output_dir, threshold=0.0, custom_min=None, custom_max=1.0):
    """
    Process images from a .npy file, apply optional thresholding, normalize, and save as PNG.
 
    Parameters:
    - npy_file_path: str, path to the .npy file containing images
    - output_dir: str, path to the directory where PNG images will be saved
    - threshold: float or None, threshold value; if None, no thresholding is applied.
    - custom_min: float or None, custom minimum value for normalization; if None, the min value is computed from the data.
    - custom_max: float or None, custom maximum value for normalization; if None, the max value is computed from the data.
    """
 
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
 
    # Compute global min and max values ignoring NaN and inf
    # data_max = np.nanmax(images_npy[np.isfinite(images_npy)])  # Maximum value ignoring NaN and inf
    # data_min = np.nanmin(images_npy[np.isfinite(images_npy)])  # Minimum value ignoring NaN and inf
 
    # Use custom_min and custom_max if provided, otherwise use computed values
    # min_val = custom_min if custom_min is not None else data_min
    # max_val = custom_max if custom_max is not None else data_max
    
    # Process each image
    for i, image in enumerate(images_npy):
        max_val = np.nanmax(image[np.isfinite(image)])  # Maximum value ignoring NaN and inf
        min_val = np.nanmin(image[np.isfinite(image)])  # Minimum value ignoring NaN and inf
        min_val = custom_min if custom_min is not None else min_val
        max_val = custom_max if custom_max is not None else max_val
 
        # Replace NaNs with min_val and infs with min_val
        image_cleaned = np.where(np.isnan(image), min_val, image)
        image_cleaned = np.where(np.isinf(image_cleaned), min_val, image_cleaned)
 
        # Apply thresholding if a threshold is provided
        if threshold is not None:
            image_cleaned = np.where(image_cleaned < threshold, min_val, image_cleaned)
 
        # Normalize the image to 0-255
        if max_val != min_val:
            image_normalized = 255 * (image_cleaned - min_val) / (max_val - min_val)
        else:
            image_normalized = np.zeros_like(image_cleaned)  # If the image has no range, use all zeros.
 
        # Convert to uint8 format for saving
        image_normalized = image_normalized.astype(np.uint8)
 
        # Stack grayscale image into 3 channels to create an RGB image
        image_rgb = np.stack([image_normalized] * 3, axis=-1)  # Convert to RGB by stacking
 
        # Generate filename with leading zeros
        filename = f'image_{(i+1):04d}.png'
 
        # Save each image as a PNG file
        image_pil = Image.fromarray(image_rgb)
        image_pil.save(os.path.join(output_dir, filename))
 
    print(f'Images saved in {output_dir}')
 
def generate_random_point_cloud(num_points=5000, 
                                                     x_range=(-1, 1), 
                                                     y_range=(-1, 1), 
                                                     z_range=(-1, 1), 
                                                     ply_file="random_point_cloud.ply", 
                                                     points3d_txt_file="points3d.txt"):
    """
    Generates a random point cloud, saves it as a binary little-endian .ply file, and also
    generates a points3d.txt file with random normals and colors.
 
    Parameters:
    - num_points (int): Number of points in the point cloud.
    - x_range (tuple): Range for X-axis values (min, max).
    - y_range (tuple): Range for Y-axis values (min, max).
    - z_range (tuple): Range for Z-axis values (min, max).
    - ply_file (str): The output file name for the .ply file.
    - points3d_txt_file (str): The output file name for the points3d.txt file.
    """
 
    # Generate random vertices (x, y, z)
    vertices = np.array([
        [random.uniform(*x_range), random.uniform(*y_range), random.uniform(*z_range)]
        for _ in range(num_points)
    ], dtype=np.float32)
 
    # Generate random normals (nx, ny, nz) and normalize them
    normals = np.random.uniform(low=-1, high=1, size=(num_points, 3)).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
 
    # Generate random colors (R, G, B) in uint8 format (0-255)
    colors = np.random.randint(0, 256, size=(num_points, 3), dtype=np.uint8)
 
    # Write to .ply file
    with open(ply_file, 'wb') as ply_file_handle:
        # Write the PLY header
        ply_header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {num_points}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float nx\n"
            "property float ny\n"
            "property float nz\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        ply_file_handle.write(ply_header.encode('ascii'))
 
        # Write to points3d.txt file
        with open(points3d_txt_file, 'w') as txt_file_handle:
            txt_file_handle.write("# 3D point list with one line of data per point:\n")
            txt_file_handle.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            txt_file_handle.write(f"# Number of points: {num_points}\n")
 
            # Combine vertices, normals, and colors and write them in binary format
            for i in range(num_points):
                # PLY data
                vertex_normal_color = struct.pack(
                    'fff'      # 3 floats for the vertex (x, y, z)
                    'fff'      # 3 floats for the normal (nx, ny, nz)
                    'BBB',     # 3 unsigned chars for the color (R, G, B)
                    vertices[i][0], vertices[i][1], vertices[i][2],
                    normals[i][0], normals[i][1], normals[i][2],
                    colors[i][0], colors[i][1], colors[i][2]
                )
                ply_file_handle.write(vertex_normal_color)
 
                # points3d.txt data
                r, g, b = colors[i]
                txt_file_handle.write(f"{i+1} {vertices[i][0]} {vertices[i][1]} {vertices[i][2]} {r} {g} {b} 0.5\n")
 
    print(f"Binary point cloud saved to {ply_file}")
    print(f"3D points saved to {points3d_txt_file}")
 
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
 
    relative_folder_path = "unprocessed/spine_phantom/left3_2/"
    folder_path = os.path.join(script_dir, relative_folder_path) # source path
 
    # recenter and load poses
    poses = recenter_poses(np.load(folder_path + "poses.npy"))
    # load and generate images
    images_npy = np.load(folder_path + "images.npy")
    print(np.shape(images_npy))
    images_path = os.path.join(folder_path, "images")
    os.makedirs(images_path, exist_ok=True)
    process_and_save_images(images_npy, images_path) # TODO check if custom max correct
 
    # define output
    output_dir = os.path.join(folder_path, "sparse/0")
    os.makedirs(output_dir, exist_ok=True)
 
    cameras_txt_path = os.path.join(output_dir, "cameras.txt")
    images_txt_path = os.path.join(output_dir, "images.txt")
    points3d_txt_path = os.path.join(output_dir, "points3D.txt")
    projection_matrix_path = os.path.join(output_dir, "orth_proj_matrix.txt")
 
    # Define camera parameters
    camera_id = 1
    model = "PINHOLE"
    height = np.shape(images_npy[0])[0]
    width = np.shape(images_npy[0])[1]
    print(f"width: {width} height: {height}")
    factor_pixels_to_mm = 0.001 # defined by dicom should be 1 pixel = 1mm?
    actual_width = width * factor_pixels_to_mm
    actual_height = height * factor_pixels_to_mm
    cx = width / 2
    cy = height / 2
 
    # near and far clipping plane (in write images we define the x translation of the camera by actual image height / 2)
    translation_1 = poses[10][:3, 3] / 1000  # Convert to meters
    translation_2 = poses[11][:3, 3] / 1000  # Convert to meters
    distance_between_images = np.linalg.norm(translation_2 - translation_1)
 
    near_plane = actual_height - (distance_between_images / 2)
    far_plane = actual_height + (distance_between_images / 2)
 
    top = translation_1[1] # top and bottom boundaries of volume # which coordinate system are we using ??? TODO maybe average over multiple image translation vectors here?
    bottom = (translation_1[1] - actual_height)
    right = (translation_1[0] - (actual_height/2) + (actual_width/2)) # left and right boundaries of volume
    left = (translation_1[0] - (actual_height/2) - (actual_width/2))
 
    tanHalfFovX = right / near_plane
    tanHalfFovY = top / near_plane
    fx = np.arctan(tanHalfFovX) * 2
    fy = np.arctan(tanHalfFovY) * 2
 
    # define orthographic projection matrix
    ortho_project_matrix = np.array([
        [(2/(right - left)), 0, 0, -((right+left)/(right-left))],
        [0, (2/(top - bottom)), 0, -((top+bottom)/(top-bottom))],
        [0, 0, (-2/(far_plane - near_plane)), -((far_plane + near_plane)/(far_plane-near_plane))],
        [0, 0, 0, 1]])
 
    with open(projection_matrix_path, 'w') as f:
        f.write("# Definition of projection matrix:\n")
        f.write("# translation_1, translation_2\n")
        f.write(f"{translation_1} {translation_2}\n")
        f.write("# distance between images:\n")
        f.write(f"{distance_between_images}\n")
        f.write("# near_plane, far_plane\n")
        f.write(f"{near_plane} {far_plane}\n")
        f.write("# actual_height, actual_width\n")
        f.write(f"{actual_height} {actual_width}\n")
        f.write("# t, b, l ,r\n")
        f.write(f"{top} {bottom} {left} {right}\n")
        f.write("# fx, fy\n")
        f.write(f"{fx} {fy}\n")
        f.write("# ortho_project_matrix\n")
        f.write(f"{ortho_project_matrix}\n")
 
    # write camera data
    write_cameras_txt(cameras_txt_path, camera_id, model, width, height, fx, fy, cx, cy)
    write_images_txt(images_txt_path, poses, camera_id, folder_path, actual_height)
 
    # GENERATE points3d.txt and points3d.ply
    ply_file_path = os.path.join(output_dir, "points3D.ply" )
    points3d_file_path = os.path.join(output_dir, "points3D.txt" )
 
    generate_random_point_cloud(num_points=30000, ply_file=ply_file_path, points3d_txt_file=points3d_file_path)
 
    # GENERATE project.ini
    ini_file_path = os.path.join(output_dir, "project.ini")
    write_colmap_ini(ini_file_path, output_dir)