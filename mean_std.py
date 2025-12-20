# read all the images in the floder and calculate mean std
import os

import cv2
import numpy as np
import pandas as pd

# path = 'C:\\Users\\User\\Desktop\\dev\\50PNG\\dem_png'

# files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and not f.endswith('.csv') and not f.endswith('025_00.png')]
# mean = 0
# std = 0
# dem_stat = 'C:\\Users\\User\\Desktop\\dev\\50PNG\\dem_png\\elevation_stats.csv'
# dem_stat = pd.read_csv(dem_stat)
# pixel_list = []
# for file_name in files:
#     image = cv2.imread(os.path.join(path, file_name), cv2.IMREAD_UNCHANGED)[:,:,0]
#     filename = file_name.split('.')[0]
#     dem_cur_state = dem_stat[dem_stat['Filename'] == int(filename)]
#     min_elev = int(dem_cur_state['Min Elevation'])
#     max_elev = int(dem_cur_state['Max Elevation'])
#         # do a normalization with max = 410 min = -3, with current max = max_elev, min = min_elev
#     real_height = image / 255 * (max_elev - min_elev) + min_elev
#     dem_image = (real_height - (-3)) / (125 + 3) * 255
#     # perform totensor()
#     dem_image = dem_image.astype(np.float32)
#     dem_image = np.clip(dem_image, 0, 255)
#     dem_image = dem_image / 255.0  # normalize to [0, 1]
#     # mean += nstd(real_height)
#     pixel_list.extend(dem_image.flatten())

# pixel_list = np.array(pixel_list)
# mean = np.mean(pixel_list)
# std = np.std(pixel_list)
# print(f'Mean: {mean}, Std: {std}')

# pixel_list = []
# path = 'C:\\Users\\User\\Desktop\\dev\\50PNG\\DEPTH_png'

# # get all the image in the path, the subfolder is 1 and one more subfolder RF01 then the png files, get all the png file name
# files = [os.path.join(path, f, rainfall, png_file) for f in os.listdir(path) for rainfall in os.listdir(os.path.join(path, f)) for png_file in os.listdir(os.path.join(path, f, rainfall)) if png_file.endswith('.png')]

# mean_sum = 0
# std_sum = 0
# pixel_count = 0

# for file in files: 
#     image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
#     if image is None:
#         continue
#     image = image.astype(np.float64) / 255.0
    
#     mean_sum += np.sum(image)
#     std_sum += np.sum(np.square(image))
#     pixel_count += image.size

# mean = mean_sum / pixel_count
# std = np.sqrt(std_sum / pixel_count - mean**2)

# print(f'Mean: {mean}, Std: {std}')

#path = 'C:\\Users\\Hank\\Desktop\\PIFF-master02\\data\\tainan_dem.png'

#file_path = [os.path.join(path, rainfall, png_file) for rainfall in os.listdir(path) for png_file in os.listdir(os.path.join(path, rainfall)) if png_file.endswith('.png')]
#mean_sum = 0
#std_sum = 0
#pixel_count = 0

#for file in file_path:
#    image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
#    if image is None:
#       continue
#    image = image.astype(np.float64) / 255.0
    
#    mean_sum += np.sum(image)
#    std_sum += np.sum(np.square(image))
#    pixel_count += image.size

#mean = mean_sum / pixel_count
#std = np.sqrt(std_sum / pixel_count - mean**2)
#print(f'Mean: {mean}, Std: {std}')

# count the percentage of the pixel value that is larger than 100
# percentage = len(np.where(pixel_list > 55)[0]) / len(pixel_list)
# print(percentage)
# print(mean, std)

# depth_folder = 'C:\\Users\\User\\Desktop\\dev\\50PNG\\DEPTH_png'
# dem_depth_folder = [f for f in os.listdir(depth_folder) if os.path.isdir(os.path.join(depth_folder, f))]

# for folder in

# Calculate mean and std for depth (d), Vx, and Vy
def calculate_mean_std(folder_path, file_pattern):
    """
    Calculate mean and std for images in a folder structure
    """
    file_paths = []
    for rainfall_folder in os.listdir(folder_path):
        rainfall_path = os.path.join(folder_path, rainfall_folder)
        if os.path.isdir(rainfall_path):
            for png_file in os.listdir(rainfall_path):
                if png_file.endswith('.png') and file_pattern in png_file:
                    file_paths.append(os.path.join(rainfall_path, png_file))
    
    mean_sum = 0
    std_sum = 0
    pixel_count = 0
    
    print(f"Processing {len(file_paths)} files for {file_pattern}...")
    
    for file_path in file_paths:
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        image = image.astype(np.float64) / 255.0
        
        mean_sum += np.sum(image)
        std_sum += np.sum(np.square(image))
        pixel_count += image.size
    
    if pixel_count > 0:
        mean = mean_sum / pixel_count
        std = np.sqrt(std_sum / pixel_count - mean**2)
        return mean, std
    else:
        return 0, 0

# Calculate for depth (d)
print("=== Calculating statistics for depth (d) ===")
depth_path = 'C:\\Users\\Hank\\Desktop\\PIFF-master02\\data\\train\\depth'
d_mean, d_std = calculate_mean_std(depth_path, '_d_')
print(f'Depth - Mean: {d_mean}, Std: {d_std}')

# Calculate for Vx
print("\n=== Calculating statistics for Vx ===")
vx_path = 'C:\\Users\\Hank\\Desktop\\PIFF-master02\\data\\train\\Vx'
vx_mean, vx_std = calculate_mean_std(vx_path, '_vx_')
print(f'Vx - Mean: {vx_mean}, Std: {vx_std}')

# Calculate for Vy
print("\n=== Calculating statistics for Vy ===")
vy_path = 'C:\\Users\\Hank\\Desktop\\PIFF-master02\\data\\train\\Vy'
vy_mean, vy_std = calculate_mean_std(vy_path, '_vy_')
print(f'Vy - Mean: {vy_mean}, Std: {vy_std}')

# Calculate for V
print("\n=== Calculating statistics for V ===")
vz_path = 'C:\\Users\\Hank\\Desktop\\PIFF-master02\\data\\train\\V'
v_mean, v_std = calculate_mean_std(vz_path, '_v_')
print(f'V - Mean: {v_mean}, Std: {v_std}')