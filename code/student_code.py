import numpy as np
# import cv2
import math


def filterImp(channel, filter):
    channel_y, channel_x = channel.shape
    sub_y, sub_x = filter.shape
    filtered_channel = np.zeros((channel_y - sub_y + 1, channel_x - sub_x + 1))

    for image_y in range(channel_y - sub_y + 1):
        for image_x in range(channel_x - sub_x + 1):
            subM = channel[image_y : image_y + sub_y, image_x : image_x + sub_x]
            subM_val = np.multiply(subM, filter)
            subM_sum = np.sum(subM_val)
            # filtered_channel[image_y+int(sub_y/2), image_x+int(sub_x/2)] = subM_sum
            filtered_channel[image_y, image_x] = subM_sum

    np.clip(filtered_channel, 0, 1)
    return filtered_channel


def my_imfilter(image, filter):
    """
    Apply a filter to an image. Return the filtered image.

    Args
    - image: numpy nd-array of dim (m, n, c)
    - filter: numpy nd-array of dim (k, k)
    Returns
    - filtered_image: numpy nd-array of dim (m, n, c)

    HINTS:
    - You may not use any libraries that do the work for you. Using numpy to work
     with matrices is fine and encouraged. Using opencv or similar to do the
     filtering for you is not allowed.
    - I encourage you to try implementing this naively first, just be aware that
     it may take an absurdly long time to run. You will need to get a function
     that takes a reasonable amount of time to run so that the TAs can verify
     your code works.
    - Remember these are RGB images, accounting for the final image dimension.
    """

    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    ############################
    ### TODO: YOUR CODE HERE ###
    pad_y = filter.shape[0] - 1
    pad_up = int(pad_y / 2)
    pad_down = pad_y - pad_up
    pad_x = filter.shape[1] - 1
    pad_left = int(pad_x / 2)
    pad_right = pad_x - pad_left
    # image = np.pad(image, ((pad_up, pad_down), (pad_left, pad_right), (0 ,0)), 'constant',
    #                     constant_values=(0, 0))
    image = np.pad(image, ((pad_up, pad_down), (pad_left, pad_right), (0, 0)), "edge")
    # print("image size after pad: ", image.shape)
    
    # cv2 is used to RGB color channel separation and merging.
    # It does not participate in the filtering process.
    # (B, G, R) = cv2.split(image)
    
    (B, G, R) = np.squeeze(np.split(image, 3, 2), 3)
    # print("B shape: ", B.shape)

    filtered_B = filterImp(B, filter)
    # print("filtered_B: ", filtered_B)
    filtered_G = filterImp(G, filter)
    # print("filtered_G: ", filtered_G)
    filtered_R = filterImp(R, filter)
    # print("filtered_R: ", filtered_R)

    # cv2 is used to RGB color channel separation and merging.
    # It does not participate in the filtering process.
    # filtered_image = cv2.merge([filtered_B, filtered_G, filtered_R])
    
    filtered_image = np.dstack((filtered_B, filtered_G, filtered_R))
    # print("filted image size: ", filtered_image.shape)
    ### END OF STUDENT CODE ####

    return filtered_image


def createGaussianKernel(kernel_size, sigma):
    radium = kernel_size // 2
    constant = 1 / (2 * math.pi * sigma**2)
    gaussian_kernel = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - radium
            y = j - radium
            gaussian_kernel[i, j] = constant * math.exp(
                -0.5 / (sigma**2) * (x**2 + y**2)
            )

    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    return gaussian_kernel

def createHighpassKernel(lowpass_kernel):
  highpass_kernel = np.zeros(lowpass_kernel.shape)
  center_x = highpass_kernel.shape[1] // 2
  center_y = highpass_kernel.shape[0] // 2
  highpass_kernel[center_y][center_x] = 1
  highpass_kernel = highpass_kernel - lowpass_kernel
  return highpass_kernel
  

def create_hybrid_image(image1, image2, filter):
    """
    Takes two images and creates a hybrid image. Returns the low
    frequency content of image1, the high frequency content of
    image 2, and the hybrid image.

    Args
    - image1: numpy nd-array of dim (m, n, c)
    - image2: numpy nd-array of dim (m, n, c)
    Returns
    - low_frequencies: numpy nd-array of dim (m, n, c)
    - high_frequencies: numpy nd-array of dim (m, n, c)
    - hybrid_image: numpy nd-array of dim (m, n, c)

    HINTS:
    - You will use your my_imfilter function in this function.
    - You can get just the high frequency content of an image by removing its low
      frequency content. Think about how to do this in mathematical terms.
    - Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
      as 'clipping'.
    - If you want to use images with different dimensions, you should resize them
      in the notebook code.
    """

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]

    ############################
    ### TODO: YOUR CODE HERE ###
    # lowpass_gaussian_kernel = createGaussianKernel(5, 3)
    lowpass_gaussian_kernel = filter
    low_frequencies = my_imfilter(image1, lowpass_gaussian_kernel)
    
    highpass_kernel = createHighpassKernel(lowpass_gaussian_kernel)
    high_frequencies = my_imfilter(image2, highpass_kernel)
    
    hybrid_image = np.clip(low_frequencies + high_frequencies, 0, 1)
    ### END OF STUDENT CODE ####
    ############################

    return low_frequencies, high_frequencies, hybrid_image
