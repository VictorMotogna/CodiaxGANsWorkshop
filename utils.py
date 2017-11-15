'''
Most parts were taken from Udacity and modified:
https://github.com/Piasy/Udacity-DLND/blob/master/face-generation/helper.py
'''

# Operating system interfaces
import os
# Python Image Library for image operations
from PIL import Image
# For math stuff
import math
# Numpy for N-dimensional array vectorized and broadcasted operations
import numpy as np

def check_for_data(data_path):
    '''
    Checks if the two datasets were placed
    in the correct folder

    :param data_path: Data folder
    '''

    # Names for pretty printing
    celeba = 'CelebA'
    mnist = 'MNIST'

    # Images folders
    celeba_folder = 'img_align_celeba'
    mnist_folder = 'mnist'

    # Images relative paths
    celeba_path = os.path.join(data_path, celeba_folder)
    mnist_path = os.path.join(data_path, mnist_folder)

    def data_check_printer(data_dir, data_name):
        if os.path.exists(data_dir):
            print('Found {} Data'.format(data_name))
        else:
            print('{} Data Not Found'.format(data_name))

    # Check if MNIST exists
    data_check_printer(celeba_path, mnist)

    # Check if Celeba exists
    data_check_printer(celeba_path, celeba)

def crop_face(image, face_area):
    """Crops face from an image"""
    start_x = (image.size[0] - face_area) // 2
    end_x = start_x + face_area

    start_y = (image.size[1] - face_area) // 2
    end_y = start_y + face_area

    cropped_image = image.crop([start_x, start_y, end_x, end_y])

    return cropped_image

def resize_image(image, width, height):
    """Resizes image to width and height"""
    return image.resize([width, height], Image.BILINEAR)

def get_image(image_path, width, height, img_mode, process_img=False):
    """
    Reads and processes image from image_path
    :param image_path: Path of image
    :param width: Width of image
    :param height: Height of image
    :param img_mode: Colour Mode of image

    :return: Image
    """

    image = Image.open(image_path)

    # MNIST images are by default at 28x28
    if image.size != (width, height) and process_img:
        # CelebA average square face area
        face_area = 108
        # Extract face area from image
        cropped_image = crop_face(image, face_area)
        # Resize image
        image = resize_image(cropped_image, width, height)

    # Convert image to given colour mode
    image = image.convert(img_mode)

    # Return numpy array
    return np.array(image)


def get_batch(img_files, width, height, img_mode, process_img=False):
    """
    Return a batch of images
    :param img_files: Path to images
    :param width: Width of image
    :param height: Height of image
    :param img_mode: Colour Mode of image

    :return: Image
    """

    # Load images in memory
    images = [get_image(img_file, width, height, img_mode, process_img) for img_file in img_files]

    # Convert to numpy for faster processing
    img_batch = np.array(images).astype(np.float32)

    # Batch needs to have 4 dimensions: (batch_size, width, height, colour_channels)
    if len(img_batch.shape) < 4:
        # Add one dimension if needed
        img_batch = img_batch.reshape(img_batch.shape + (1,))

    return img_batch

# Thanks to math from udacity #
def images_to_grid(images, mode):
    """
    Save images as a square grid
    :param images: Images to be used for the grid
    :param mode: The mode to use for images
    :return: Image of images in a square grid
    """
    # Get maximum size for square grid of images
    save_size = math.floor(np.sqrt(images.shape[0]))

    # Scale to 0-255
    images = (((images - images.min()) * 255) / (images.max() - images.min())).astype(np.uint8)

    # Put images in a square arrangement
    images_in_square = np.reshape(
            images[:save_size*save_size],
            (save_size, save_size, images.shape[1], images.shape[2], images.shape[3]))
    if mode == 'L':
        images_in_square = np.squeeze(images_in_square, 4)

    # Combine images to grid image
    new_im = Image.new(mode, (images.shape[1] * save_size, images.shape[2] * save_size))
    for col_i, col_images in enumerate(images_in_square):
        for image_i, image in enumerate(col_images):
            im = Image.fromarray(image, mode)
            new_im.paste(im, (col_i * images.shape[1], image_i * images.shape[2]))

    return new_im


class Dataset(object):
    """
    Dataset
    """
    def __init__(self, dataset_name, data_files):
        """
        Initalize the class
        :param dataset_name: Database name
        :param data_files: List of files in the database
        """
        DATASET_CELEBA_NAME = 'celeba'
        DATASET_MNIST_NAME = 'mnist'
        IMAGE_WIDTH = 28
        IMAGE_HEIGHT = 28

        self.process_img = True if dataset_name == DATASET_CELEBA_NAME else False

        if dataset_name == DATASET_CELEBA_NAME:
            self.image_mode = 'RGB'
            image_channels = 3

        elif dataset_name == DATASET_MNIST_NAME:
            self.image_mode = 'L'
            image_channels = 1

        self.data_files = data_files
        self.shape = len(data_files), IMAGE_WIDTH, IMAGE_HEIGHT, image_channels

    def get_batches(self, batch_size):
        """
        Generate batches
        :param batch_size: Batch Size
        :return: Batches of data
        """
        IMAGE_MAX_VALUE = 255

        current_index = 0
        while current_index + batch_size <= self.shape[0]:
            data_batch = get_batch(
                self.data_files[current_index:current_index + batch_size],
                *self.shape[1:3],
                self.image_mode, self.process_img)

            current_index += batch_size

            yield data_batch / IMAGE_MAX_VALUE - 0.5
