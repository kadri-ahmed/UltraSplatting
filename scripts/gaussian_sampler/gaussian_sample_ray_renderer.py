import torch
import time
import matplotlib.pyplot as plt
import numpy as np

def render_2D(images, width, height):
    def update_image(index):
        ax.clear()
        ax.imshow(np.flipud(images[index]), cmap='gray', extent=[0, width, 0, height])
        ax.set_title(f'3D Gaussians Sampled Along Parallel Rays - Slice {index}')
        ax.set_xlabel('Ray index')
        ax.set_ylabel('Sample index')
        plt.draw()

    def on_scroll(event):
        if event.button == 'up':
            current_index[0] = (current_index[0] + 10) % num_images
        elif event.button == 'down':
            current_index[0] = (current_index[0] - 10) % num_images
        update_image(current_index[0])

    num_images = len(images)
    fig, ax = plt.subplots()
    current_index = [int(num_images/2)]
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    update_image(int(num_images/2))
    plt.show()

def gaussian_3d(pos, mean, cov):
    """Compute the value of a 3D Gaussian at positions."""
    mean = torch.tensor(mean, dtype=torch.float32, device=device)
    cov = torch.tensor(cov, dtype=torch.float32, device=device)
    inv_cov = torch.inverse(cov)
    norm_factor = (2 * torch.pi) ** (-1.5) * torch.det(cov) ** (-0.5)
    diff = pos - mean
    exponent = -0.5 * (diff @ inv_cov * diff).sum(dim=-1)
    return norm_factor * torch.exp(exponent)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# this computation time scales fairly linearly per gaussian (possibly exponentially per image)
# these are 30 gaussians we define
# 30    - computation time : 0.20191290000002482
# 300   - computation time : 1.0526036999999633
# 3000  - computation time : 9.550902899999983
# 30000 - computation time : 95.01169310000023
# for 5.000.000 sample points within the images below

# we have 213.200 
gaussians = [
    {'mean': [-2, 0, -5], 'cov': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
    {'mean': [0, 0, -10], 'cov': [[1, 0, 0], [0, 2, 0], [0, 0, 1]]},
    {'mean': [0, 0, 0], 'cov': [[0.5, 0, 0.5], [0.5, 1, 0], [0, 0.5, 1]]},
    {'mean': [1, 2, 3], 'cov': [[1, 0.2, 0], [0.2, 1, 0.2], [0, 0.2, 1]]},
    {'mean': [-3, 1, 2], 'cov': [[2, 0.5, 0], [0.5, 2, 0.5], [0, 0.5, 2]]},
    {'mean': [4, -1, -2], 'cov': [[1, 0.1, 0.1], [0.1, 1, 0.1], [0.1, 0.1, 1]]},
    {'mean': [-5, -5, -5], 'cov': [[3, 1, 0], [1, 3, 1], [0, 1, 3]]},
    {'mean': [6, 0, -1], 'cov': [[2, 0, 0.5], [0, 2, 0.5], [0.5, 0.5, 2]]},
    {'mean': [2, -3, 4], 'cov': [[1.5, 0.3, 0], [0.3, 1.5, 0.3], [0, 0.3, 1.5]]},
    {'mean': [-1, 1, 0], 'cov': [[1, 0.4, 0], [0.4, 1, 0.4], [0, 0.4, 1]]},
    {'mean': [3, 5, -2], 'cov': [[2.5, 0.5, 0.5], [0.5, 2.5, 0.5], [0.5, 0.5, 2.5]]},
    {'mean': [0, -2, 2], 'cov': [[1, 0.6, 0.6], [0.6, 1, 0.6], [0.6, 0.6, 1]]},
    {'mean': [7, 3, 1], 'cov': [[3, 0.8, 0.8], [0.8, 3, 0.8], [0.8, 0.8, 3]]},
    {'mean': [-4, -1, 5], 'cov': [[2, 0.7, 0], [0.7, 2, 0.7], [0, 0.7, 2]]},
    {'mean': [0, 0, 0], 'cov': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
    {'mean': [8, -3, -4], 'cov': [[4, 1, 0.5], [1, 4, 0.5], [0.5, 0.5, 4]]},
    {'mean': [5, 5, 5], 'cov': [[2, 0.5, 0.5], [0.5, 2, 0.5], [0.5, 0.5, 2]]},
    {'mean': [-2, -2, -2], 'cov': [[1, 0.2, 0.2], [0.2, 1, 0.2], [0.2, 0.2, 1]]},
    {'mean': [4, 2, -1], 'cov': [[1.5, 0.3, 0.3], [0.3, 1.5, 0.3], [0.3, 0.3, 1.5]]},
    {'mean': [-7, 1, 0], 'cov': [[3, 1, 1], [1, 3, 1], [1, 1, 3]]},
    {'mean': [6, -5, 3], 'cov': [[2.5, 0.7, 0.7], [0.7, 2.5, 0.7], [0.7, 0.7, 2.5]]},
    {'mean': [0, 2, -3], 'cov': [[1, 0.4, 0.4], [0.4, 1, 0.4], [0.4, 0.4, 1]]},
    {'mean': [3, -4, 1], 'cov': [[2, 0.6, 0.6], [0.6, 2, 0.6], [0.6, 0.6, 2]]},
    {'mean': [-6, 0, 5], 'cov': [[3, 0.8, 0.8], [0.8, 3, 0.8], [0.8, 0.8, 3]]},
    {'mean': [5, 3, -2], 'cov': [[2, 0.5, 0.5], [0.5, 2, 0.5], [0.5, 0.5, 2]]},
    {'mean': [-3, 2, -4], 'cov': [[1, 0.3, 0.3], [0.3, 1, 0.3], [0.3, 0.3, 1]]},
    {'mean': [2, 1, 1], 'cov': [[1.5, 0.2, 0.2], [0.2, 1.5, 0.2], [0.2, 0.2, 1.5]]},
    {'mean': [0, -4, 3], 'cov': [[2, 0.7, 0.7], [0.7, 2, 0.7], [0.7, 0.7, 2]]},
    {'mean': [4, 0, -5], 'cov': [[2.5, 0.6, 0.6], [0.6, 2.5, 0.6], [0.6, 0.6, 2.5]]},
    {'mean': [-2, 3, 2], 'cov': [[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]]},
]

# you can extend the list here for more gaussians
gaussians = gaussians * 10

# define the central origin and direction for the rays
origin = torch.tensor([0, 0, -10], dtype=torch.float32, device=device)
direction = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)

# image parameters
image_width = 100
image_height = 100
num_images = 500
samples_per_ray = image_height
step_size = 0.1
ray_spacing = 0.1

all_images = torch.zeros((num_images, image_height, image_width), dtype=torch.float32, device=device)

# generate rays and sample positions
# am i doing this right? i think there is an error somewhere? but it might be in how i think about this entirely
ray_offsets = torch.arange(image_width, dtype=torch.float32, device=device) * ray_spacing - (image_width * ray_spacing / 2)
ray_origins = origin.unsqueeze(0) + torch.stack([ray_offsets, torch.zeros(image_width, dtype=torch.float32, device=device), torch.zeros(image_width, dtype=torch.float32, device=device)], dim=1)

sample_indices = torch.arange(samples_per_ray, dtype=torch.float32, device=device).unsqueeze(1).unsqueeze(2)
sample_positions = ray_origins.unsqueeze(0) + sample_indices * direction.view(1, 1, -1) * step_size

start = time.perf_counter()

# vectorize the z-offsets for all images
z_offsets = (torch.arange(num_images, dtype=torch.float32, device=device) * step_size - (0.5 * num_images * step_size)).view(-1, 1, 1, 1)

# add the z-offsets to the sample positions
all_sample_positions = sample_positions.unsqueeze(0) + torch.tensor([0, 0, 1], dtype=torch.float32, device=device) * z_offsets

# flatten the sample positions for batch processing
flat_sample_positions = all_sample_positions.view(-1, 3)

# TODO we can still improve compute times here
# compute the Gaussian values for each sample position
total_values = torch.zeros(flat_sample_positions.shape[0], dtype=torch.float32, device=device)
for gaussian in gaussians:
    total_values += gaussian_3d(flat_sample_positions, gaussian['mean'], gaussian['cov'])

# reshape the results back to the image dimensions
all_images = total_values.view(num_images, samples_per_ray, image_width)

end = time.perf_counter()
print(f"computation time: {end - start}")

all_images_cpu = all_images.cpu().numpy()
render_2D(all_images_cpu, image_width, image_height)
