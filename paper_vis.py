import cv2
import numpy as np


def contrast_stretching(img):
    min = np.min(img)
    max = np.max(img)
    img = (img - 200) / (255 - 200) * 255
    img = img.astype(np.uint8)
    return img

image_path = 'C:\\Users\\User\\Desktop\\dev\\I2SB-flood\\results\\flood-dems-otode-b16-mask\\test3_nfe10\\recon_61_RF08_024.png'
# image_path = 'C:\\Users\\User\\Desktop\\dev\\new_test\\TEST_png\\RF133\\RF133_d_020_00.png'

image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)[:,:,0]

image_test_path = 'C:\\Users\\User\\Desktop\\dev\\test_dem\\61\\RF08\\61_RF08_024.png'
image_test = cv2.imread(image_test_path, cv2.IMREAD_UNCHANGED)

diff_image = cv2.absdiff(image, image_test)

dem_image_path = 'C:\\Users\\User\\Desktop\\dev\\50PNG\\dem_png\\61.png'
dem = cv2.imread(dem_image_path, cv2.IMREAD_UNCHANGED)[:,:,0]

# zeros = np.where(image == 0)
# print("Zero-value pixels:", zeros)

image = contrast_stretching(image)

# cv2.imwrite('contrast_133_20_gen_nfe1.png', image)

# upper_left = (100, 20)
# lower_right = (200, 120)
upper_left = (30,20)
lower_right = (130, 120)

crop = image[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]

# # save the crop 
cv2.imwrite('crop_gen_nfe1_crop2.png', crop)


# test_rainfall = 'C:\\Users\\User\\Desktop\\dev\\new_test\\test.csv'

# # plot the column of inflow_132 as time series plot
# import matplotlib.pyplot as plt
# import pandas as pd

# time = np.arange(1, 25)
# rainfall = pd.read_csv(test_rainfall)
# cur_rainfall = rainfall['inflow_133'][1:]
# fig = plt.figure(figsize=(8,4))
# plt.plot(cur_rainfall)
# # time = time.flatten()
# # cur_rainfall = cur_rainfall.flatten()
# # plt.bar(time, cur_rainfall, color='b')
# plt.scatter(time, cur_rainfall, color='b')
# plt.xticks(time, fontsize=14)
# plt.yticks(fontsize=14)
# plt.xlabel('Rainfall Timestep(H)', fontsize=18)
# plt.ylabel('Rainfall Intensity(mm/hr)', fontsize=18)
# # plt.title('Rainfall Intensity for Virtual Inspection', fontsize=14)
# plt.tight_layout()
# plt.savefig('rainfall_132.png')
