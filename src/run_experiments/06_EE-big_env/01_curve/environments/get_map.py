import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

curr_dir = os.path.dirname(os.path.abspath(__file__))

# Load the image
image_path = os.path.join(curr_dir, 'track1.png')
image = Image.open(image_path)

# Resize the image to 2000x2000
image_resized = image.resize((200, 200), Image.LANCZOS)

# Convert image to grayscale
image_gray = image_resized.convert('L')

# Convert grayscale image to numpy array
image_array = np.array(image_gray)

# Threshold the image to get a binary representation of the track: track (0), non-track (1)
# We assume the track is darker than the surrounding area, so we use a lower threshold

threshold = 200  # The value might need adjustments after examining the image
track_array = np.where(image_array < threshold, 0, 1)

# Display a small part of the array as a sanity check
plt.imshow(track_array, cmap='gray')
plt.show()

# Return the shape of the array to verify size
track_array.shape

# Convert the numpy array to a string representation
# Ones will be represented as hashtags (#) and zeros as dots (.)

def array_to_string(arr):
    # Map the values in the array to characters, joining them with no spaces
    return '\n'.join(''.join('#' if cell else '.' for cell in row) for row in arr)

# Call the function to convert our array to a string representation
track_string = array_to_string(track_array)
# Define the file path for the text file
text_file_path = os.path.join(curr_dir, 'track_string.txt')

# Write the track string to the text file
with open(text_file_path, 'w') as file:
    file.write(track_string)

# Print a message to confirm that the track string has been written to the text file
print("Track string has been written to the text file:", text_file_path)

# Due to the large size of the matrix, we won't print the entire string here.
# Instead, we'll verify by printing just the first 100 characters in the first line

#print(track_string)

