# Adding-Sunglasses-to-Your-Passport-Photo-Using-OpenCV
# Name: Sanchita Sandeep
# Reg No: 212224240142
## Aim
To digitally overlay a transparent PNG image of sunglasses onto a person's face image at the correct position using alpha channel masking and NumPy array slicing/arithmetic blending operations in OpenCV.
## Software Used
Anaconda – Python 3.7 Jupyter Notebook / VS Code OpenCV (cv2)
## Algorithm
```
1)Import Libraries: Import cv2, numpy, and matplotlib.pyplot.
2)Load Images: Read the base face image and the transparent sunglasses PNG (-1 flag).
3)Resize Overlay: Scale glassPNG to match the target eye region dimensions (670 x 200).
4)Split Channels: Extract the BGR color channels (glassBGR) and the alpha transparency mask (glassMask1).
5)Define ROI: Set the slice coordinate boundaries on the face image ([530:730, 210:880]).
6)Normalize Mask: Convert the single-channel mask into a 3-channel format and scale values from [0, 255] to [0, 1].
7)Arithmetic Blending: Extract the target eyeROI from a copy of the face image.
8)Add the masked eye and masked glass regions together (eyeRoiFinal).
9)Final Placement & Display: Insert eyeRoiFinal back into the face image at the defined ROI coordinates and display the result.
```
## Program
```
# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
faceImage = cv2.imread('3E108C11-46EF-4E1E-A24B-ECB3E19B3DCA.png')
plt.imshow(faceImage[:,:,::-1]);plt.title("Face")
faceImage.shape
#resized_faceImage.shape
faceImage.shape
# Load the Sunglass image with Alpha channel
# (http://pluspng.com/sunglass-png-1104.html)
glassPNG = cv2.imread('sunglass-png-aviator-20sunglasses-20png-265.png',-1)
plt.imshow(glassPNG[:,:,::-1]);plt.title("glassPNG")
# Resize the sunglasses to better match the width and height of her actual glasses
glassPNG = cv2.resize(glassPNG, (670, 200))
print("image Dimension ={}".format(glassPNG.shape))
glassBGR = glassPNG[:,:,0:3]
glassMask1 = glassPNG[:,:,3]
plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(glassBGR[:,:,::-1]);plt.title('Sunglass Color channels');
plt.subplot(122);plt.imshow(glassMask1,cmap='gray');plt.title('Sunglass Alpha channel');
# Make a copy
#faceWithGlassesNaive = resized_faceImage.copy()
faceWithGlassesNaive = faceImage.copy()
# Replace the eye region with the sunglass image
faceWithGlassesNaive[530:730, 210:880]=glassBGR
plt.figure(figsize = [10, 10])
plt.imshow(faceWithGlassesNaive[...,::-1])
# Make the dimensions of the mask same as the input image.
glassMask = cv2.merge((glassMask1, glassMask1, glassMask1)) 
# Make the values [0,1] since we are using arithmetic operations
glassMask = np.uint8(glassMask / 255) 
# Make a copy
faceWithGlassesArithmetic = faceImage.copy() 
# Get the eye region from the face image using the correct bounding box
eyeROI = faceWithGlassesArithmetic[530:730, 210:880] 
# Use the mask to create the masked eye region
maskedEye = cv2.multiply(eyeROI, (1 - glassMask)) 
# Use the mask to create the masked sunglass region
maskedGlass = cv2.multiply(glassBGR, glassMask) 
# Combine the Sunglass in the Eye Region to get the augmented image
eyeRoiFinal = cv2.add(maskedEye, maskedGlass) 
# Display the intermediate results
plt.figure(figsize = [20, 20])
plt.subplot(131); plt.imshow(maskedEye[..., ::-1]); plt.title("Masked Eye Region")
plt.subplot(132); plt.imshow(maskedGlass[..., ::-1]); plt.title("Masked Sunglass Region")
plt.subplot(133); plt.imshow(eyeRoiFinal[..., ::-1]); plt.title("Augmented Eye and Sunglass")
# Replace the eye ROI with the output from the previous section
# Replace the eye ROI with the output from the previous section
faceWithGlassesArithmetic[530:730, 210:880] = eyeRoiFinal 
# Display the final result
plt.figure(figsize = [20, 20])
plt.subplot(121); plt.imshow(faceImage[:, :, ::-1]); plt.title("Original Image")
plt.subplot(122); plt.imshow(faceWithGlassesArithmetic[:, :, ::-1]); plt.title("With Sunglasses")
```
## Output

1) Loading Face Image
<img width="342" height="434" alt="download" src="https://github.com/user-attachments/assets/4e2843e8-6638-4fca-83be-a3b6e7fae9ae" />


2) Load the Sunglass image with Alpha channel
<img width="546" height="434" alt="download" src="https://github.com/user-attachments/assets/0df0acdc-259e-4e4d-9d2b-2ff866454a6d" />


3) Separate the Color and alpha channels
<img width="1219" height="222" alt="download" src="https://github.com/user-attachments/assets/e0e74d0a-fadf-409f-81ee-dec1e4d6c285" />


4)Replace the eye region with the sunglass image
<img width="643" height="818" alt="download" src="https://github.com/user-attachments/assets/a5a36dc1-e75e-432b-aed0-d7d68fd12795" />


5)Combine the Sunglass in the Eye Region to get the augmented image
<img width="1606" height="201" alt="download" src="https://github.com/user-attachments/assets/13e42c09-3f2d-4d33-a26a-36c0478f8405" />


6)Display the final result
<img width="1615" height="1004" alt="download" src="https://github.com/user-attachments/assets/fd9407fd-6725-4bc3-bafd-92d205c597c2" />

## Result: 
The sunglasses have been successfully place on the person's eye
















