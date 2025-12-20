import os
import cv2
import numpy as np
from PIL import Image
from scipy.spatial import distance
from pytorch_fid import fid_score
import torch

def water_depth_max_calculation(real_image, fake_image):
    # find the index of the largest value in the real image
    smallest_index = np.unravel_index(np.argmin(real_image, axis=None), real_image.shape)
    # find the value of the largest value in the real image
    smallest_value = real_image[smallest_index]
    # find the value of the largest value in the fake image
    fake_smallest_value = fake_image[smallest_index]

    diff_largest = abs((smallest_value - fake_smallest_value) / 255 * 4)

    return diff_largest 

def calculate_distances(real_folder, fake_folder):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    real_images = []
    fake_images = []

    # Read real images
    real_filenames = sorted([filename for filename in os.listdir(real_folder) if filename.endswith(".jpg") or filename.endswith(".png")])
    fake_filenames = sorted([filename for filename in os.listdir(fake_folder) if filename.endswith(".jpg") or filename.endswith(".png")])

    # Read real images
    for filename in real_filenames:
        filename = os.path.join(real_folder, filename)
        flood_image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        # remove the fourth channel
        # flood_image = cv2.cvtColor(flood_image, cv2.COLOR_BGR2GRAY)
        real_images.append(np.array(flood_image, dtype=np.float32))

    # Read fake images
    for filename in fake_filenames:
        image1 = Image.open(os.path.join(fake_folder, filename)).convert('L')
        fake_images.append(np.array(image1, dtype=np.float32))

    # Calculate distances
    l1_distances = []
    water_l1_depth = []
    water_rmse_depth = []
    max_water_depth = []
    water_percentage_depth = []
    l2_distances = []
    linf_distances = []

    for real_image, fake_image in zip(real_images, fake_images):
        l1_mean = np.mean(abs(real_image - fake_image))
        l1_distances.append(l1_mean)
        water_l1_depth.append(round(l1_mean/255*4,2))
        l2_mean = np.sqrt(np.mean((real_image - fake_image) ** 2))
        l2_distances.append(l2_mean)
        water_rmse_depth.append(l2_mean/255*4)
        water_percentage_depth.append(l1_mean/255)
        max_water_depth.append(water_depth_max_calculation(real_image, fake_image))
        # l2_distances.append(np.sqrt(np.mean((real_image - fake_image) ** 2)))
        linf_distances.append(np.max(abs(real_image - fake_image)))

    # print filename with l1 distances
    # for filename, l1_distance in zip(real_filenames, l1_distances):
    #     print(filename, l1_distance)
    pathes = [real_folder, fake_folder]
    # Calculate FID
    fid = fid_score.calculate_fid_given_paths(pathes, 64, device, 2048, num_workers=1)
    
    return real_filenames, fid, l1_distances, l2_distances, linf_distances, water_l1_depth, water_rmse_depth, max_water_depth, water_percentage_depth
    
    # return avg_l1_distance, avg_l2_distance, avg_linf_distance, fid, avg_water_depth, avg_water_rmse_depth, avg_max_water_depth, avg_water_percentage_depth
import os
import pandas as pd 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
if __name__ == "__main__":
# Usage example
# real_folder = "/home/Josh/BrightestImageProcessing/Josh/image_generation/style_transfer/pytorch-CycleGAN-and-pix2pix/datasets/euv/image/test"
# fake_folder = "/home/Josh/BrightestImageProcessing/Josh/image_generation/style_transfer/pytorch-CycleGAN-and-pix2pix/results/p2p_wgangp_bs64_batch_pixel"
    real_folder = "C:\\Users\\Hank\\Desktop\\dev\\new_test\\test_total"
    fake_folder = "C:\\Users\\Hank\\Desktop\\dev\\I2SB-flood\\results\\flood-latent-new-b4\\test3_nfe1"
    saving_path_name = 'latent_i2sb'
    real_filenames, fid, l1_distances, l2_distances, linf_distances, water_l1_depth, water_rmse_depth, max_water_depth, water_percentage_depth = calculate_distances(real_folder, fake_folder)

    # save l1_distances, l2_distances, linf_distances, water_depth, water_rmse_depth, max_water_depth, water_percentage_depth to a csv file
    distances_pd = pd.DataFrame(list(zip(real_filenames, l1_distances, l2_distances, linf_distances, water_l1_depth, water_rmse_depth, max_water_depth, water_percentage_depth)), 
                                columns = ['filenames', 'L1 Distance', 'L2 Distance', 'L-infinity Distance', 'Water L1 Depth', 'Water RMSE Depth', 'Max Water Depth', 'Water Percentage Depth'])
    saving_file_name = f"{saving_path_name}.csv"
    distances_pd.to_csv("C:\\Users\\Hank\\Desktop\\dev\\distance_result\\" + saving_file_name)

    avg_l1_distance = np.mean(l1_distances)
    avg_l2_distance = np.mean(l2_distances)
    avg_linf_distance = np.mean(linf_distances)
    avg_water_depth = np.mean(water_l1_depth)
    avg_water_rmse_depth = np.mean(water_rmse_depth)
    avg_max_water_depth = np.mean(max_water_depth)
    avg_water_percentage_depth = np.mean(water_percentage_depth)

    print("Average L1 distance:", avg_l1_distance)
    print("Average L2 distance:", avg_l2_distance)
    print("Average L-infinity distance:", avg_linf_distance)
    print("FID:", fid)
    print("Average Water Depth:", avg_water_depth)
    print("Average Water RMSE Depth:", avg_water_rmse_depth)
    print("Average Max Water Depth:", avg_max_water_depth)
    print("Average Water Percentage Depth:", avg_water_percentage_depth)
    print("\n")

    # calculate the metrics for the first 50 rows, 50-125 rows, and 125-500
    dis = [0, 50, 125, 500]
    for index,item in enumerate(dis):
        if index == 0:
            continue
        average_l1_distance = np.mean(l1_distances[dis[index-1]:dis[index]])
        average_l2_distance = np.mean(l2_distances[dis[index-1]:dis[index]])
        average_linf_distance = np.mean(linf_distances[dis[index-1]:dis[index]])  
        average_water_depth = np.mean(water_l1_depth[dis[index-1]:dis[index]])
        average_water_rmse_depth = np.mean(water_rmse_depth[dis[index-1]:dis[index]])
        average_max_water_depth = np.mean(max_water_depth[dis[index-1]:dis[index]])
        average_water_percentage_depth = np.mean(water_percentage_depth[dis[index-1]:dis[index]])
        print(f"Average L1 distance for {item} rows:", average_l1_distance)
        print(f"Average L2 distance for {item} rows:", average_l2_distance)
        print(f"Average L-infinity distance for {item} rows:", average_linf_distance)
        print(f"Average Water Depth for {item} rows:", average_water_depth)
        print(f"Average Water RMSE Depth for {item} rows:", average_water_rmse_depth)
        print(f"Average Max Water Depth for {item} rows:", average_max_water_depth)
        print(f"Average Water Percentage Depth for {item} rows:", average_water_percentage_depth)
        print("\n")






