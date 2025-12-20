import os
import cv2
import pandas as pd

def extract_parts(filename):
    parts = filename.split("_")
    return int(parts[0][2:]), int(parts[2])  # Returns '03' and '025'

rainfall_path = 'C:\\Users\\Hank\\Desktop\\dev\\PNG_TUFLOW\\scenario_rainfall.csv'
rainfall = pd.read_csv(rainfall_path)
# remove first row, no 0 row 
rainfall = rainfall.iloc[1:, 1:]

# Initialize lists to store cell values and their positions
rainfall_cum_value = []
cell_positions = []

val = False
# Iterate through each column
for col in rainfall.columns:
    col_num = int(col.split("_")[1])
    if (val and col_num not in [5, 13, 16, 26, 36, 46]) or (not val and col_num in [5, 13, 16, 26, 36, 46]):
        continue
    cell_values = []
    # Iterate through each row in the current column
    for row in range(len(rainfall)):
        cell_value = rainfall.iloc[row][col]
        cell_values.append(cell_value)
        # make it a len 24 list if not append 0 in front
        temp = [0] * (24 - len(cell_values))
        temp.extend(cell_values)
        rainfall_cum_value.append(temp)
        cell_positions.append((col_num, row+1))

# Function to find the corresponding image in the flood path folder

# random pick 10 rainfall_cum_value and cell_position to test
print(rainfall_cum_value[0:10])
print(cell_positions[0:10])
print(len(rainfall_cum_value))

flood_path = 'C:\\Users\\Hank\\Desktop\\dev\\PNG_TUFLOW\\tainan_png'

def find_image(cell_position, flood_path):
    col, row = cell_position
    folder_name = f"RF{col:02d}"
    image_name = f"{folder_name}_d_{row:03d}_00.png"
    image_path = os.path.join(flood_path, folder_name, image_name)
    return image_path

# Example usage
index = 5
example_position = cell_positions[index]
image_path = find_image(example_position, flood_path)
print(example_position, image_path)

index = 8
example_position = cell_positions[index]
image_path = find_image(example_position, flood_path)
print(example_position, image_path)

index = 489
example_position = cell_positions[index]
image_path = find_image(example_position, flood_path)
print(example_position, image_path)
print(rainfall_cum_value[index])

index = 7
example_position = cell_positions[index]
image_path = find_image(example_position, flood_path)
print(example_position, image_path)
print(rainfall_cum_value[index])
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
print(image.shape)
cv2.imshow("Image", image)
cv2.waitKey(0)
print(image[0, 0])
# val 5 13 16 26 36 46
# train 1 3 6 8 10 11 12 14 15 17 18 19 20 21 22 23 24 25 27 28 29 30 31 32 33 34 35 37 38 39 40 41 42 43 44 45 47 48 49 50


