import cv2
import iris
import numpy as np
import matplotlib.pyplot as plt

def calculate_masked_fractional_hd(iris_code1, iris_code2, mask1, mask2):
    # Flatten the iris codes to compare bit by bit
    flat_code1 = iris_code1.flatten()
    flat_code2 = iris_code2.flatten()

    # Apply the masks to both iris codes
    masked_code1 = flat_code1 & mask1.flatten()
    masked_code2 = flat_code2 & mask2.flatten()

    # Calculate the Hamming distance
    hamming_distance = np.sum(masked_code1 != masked_code2)

    # Calculate the masked fractional Hamming Distance
    frac = (np.sum(mask1.flatten()) + np.sum(mask2.flatten()))/2
    masked_fractional_hd = hamming_distance / frac

    return masked_fractional_hd

iris_pipeline = iris.IRISPipeline(env=iris.IRISPipeline.DEBUGGING_ENVIRONMENT)
iris_visualizer = iris.visualisation.IRISVisualizer()

# Read the label file
with open("label.txt", "r") as txt:
    lines = txt.readlines()

data = r'data\\'

img_path_1  = data + lines[0][:-1]
img_pixels_1 = cv2.imread(img_path_1, cv2.IMREAD_GRAYSCALE)
output_1 = iris_pipeline(img_data=img_pixels_1, eye_side="left")
for i in range(len(lines)):
    img_path_2 = data + lines[i][:-1]
    img_pixels_2 = cv2.imread(img_path_2, cv2.IMREAD_GRAYSCALE)
    output_2 = iris_pipeline(img_data=img_pixels_2, eye_side="left")
    # print(output_1['error'])
    hd = calculate_masked_fractional_hd(output_1['iris_template']['iris_codes'], output_2['iris_template']['iris_codes'],
                                             output_1['iris_template']['mask_codes'], output_2['iris_template']['mask_codes'])
    print(lines[i]," ", hd)
        
    
"""
# Initialize a heatmap
heatmap = np.zeros((len(lines), len(lines)))

# Iterate over image pairs and calculate Hamming distances
for i in range(len(lines)):
    for j in range(i+1, len(lines)):
        img_path_1, img_path_2 = data + lines[i][:-1], data + lines[j][:-1]
        img_pixels_1 = cv2.imread(img_path_1, cv2.IMREAD_GRAYSCALE)
        img_pixels_2 = cv2.imread(img_path_2, cv2.IMREAD_GRAYSCALE)
        output_1 = iris_pipeline(img_data=img_pixels_1, eye_side="left")
        output_2 = iris_pipeline(img_data=img_pixels_2, eye_side="left")
        # print(output_1['error'])
        hd = calculate_masked_fractional_hd(output_1['iris_template']['iris_codes'], output_2['iris_template']['iris_codes'],
                                             output_1['iris_template']['mask_codes'], output_2['iris_template']['mask_codes'])
        heatmap[i, j] = hd
        heatmap[j, i] = hd
        # canvas = iris_visualizer.plot_ir_image(iris.IRImage(img_data=img_pixels_1, eye_side="left"))
        # canvas = iris_visualizer.plot_segmentation_map(output_1['segmentation_map'])
        canvas = iris_visualizer.plot_iris_template(output_1['iris_template'])
        plt.show()
        break
    break
        # plt.show()
# Plot the heatmap
# plt.figure(figsize=(10, 10))
# plt.imshow(heatmap, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.show()
"""