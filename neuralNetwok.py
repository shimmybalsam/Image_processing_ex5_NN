import numpy as np
from scipy.misc import imread
from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, Add
from tensorflow.keras.optimizers import Adam
from . import sol5_utils

MAX_PIX_LEVEL = 255
GRAYSCALE = 1
COLOR = 2
DIM_BORDER = 32
RGB_CHANNELS = 3
ZERO_MEAN = 0
EIGHTY_PERCENT_FACTOR = 0.8
CROP_SIZE_MULTIPLYER = 3


def read_image(filename, representation):
    """
    opens image as matrix
    :param filename: name of image
    :param representation: 1 if grayscale, 2 if RGB
    :return: np.array float64 of given image
    """
    image = imread(filename)
    image_float = image.astype(np.float64) / MAX_PIX_LEVEL
    if representation == GRAYSCALE:
        image_float_gray = rgb2gray(image_float)
        return image_float_gray
    return image_float


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    loads dtasets
    :param filenames: images names
    :param batch_size: size of batch to run on during each epoc
    :param corruption_func: how to currupt a testing picture to train on
    :param crop_size: shape of patch
    :return: generator, a tuple consisting of list source bathces and list of target batches
    """
    height, width = crop_size[0], crop_size[1]
    source_batch = np.zeros((batch_size,height,width,1))
    target_batch = np.zeros((batch_size,height,width,1))
    cache = {}
    while True:

        for i in range(batch_size):
            filename = np.random.choice(filenames)
            if filename in cache.keys():
                im = cache[filename]
            else:
                im = read_image(filename,GRAYSCALE)
                cache[filename] = im
            y, x = im.shape
            begin_crop_x = np.random.randint(x - width*CROP_SIZE_MULTIPLYER)
            begin_crop_y = np.random.randint(y - height*CROP_SIZE_MULTIPLYER)
            cropped = im[begin_crop_y:begin_crop_y+height*CROP_SIZE_MULTIPLYER, begin_crop_x:begin_crop_x+width*CROP_SIZE_MULTIPLYER]
            smaller_cropped = cropped[:height,:width]
            target_batch[i] = smaller_cropped.reshape((smaller_cropped.shape[0], smaller_cropped.shape[1], 1)) - 0.5
            corrupted_crop = corruption_func(cropped)
            smaller_corrupted_crop = corrupted_crop[:height,:width]
            source_batch[i] = smaller_corrupted_crop.reshape((smaller_corrupted_crop.shape[0], smaller_corrupted_crop.shape[1], 1)) - 0.5

        yield (source_batch, target_batch)


def resblock(input_tensor, num_channels):
    """
    building one residual block of network
    :param input_tensor: start inputs of block
    :param num_channels: amount of channels in image
    :return: the solution calculated by block after convolution and relu etc.
    """
    output = Conv2D(num_channels, (3,3),padding='same')(input_tensor)
    output = Activation('relu')(output)
    output = Conv2D(num_channels, (3, 3), padding='same')(output)
    output = Add()([input_tensor, output])
    output = Activation('relu')(output)
    return output


def build_nn_model(height, width, num_channels, num_res_blocks):
    """
    building an entire NN network
    :param height: patch height
    :param width: patch width
    :param num_channels: amount of channels in image
    :param num_res_blocks: amount of residual blocks in the entire network
    :return: A full NN which can be trained on
    """
    initial = Input(shape=(height, width, 1))
    next_in_model = Conv2D(num_channels, (3,3),padding='same')(initial)
    next_in_model = Activation('relu')(next_in_model)
    for i in range(num_res_blocks):
        next_in_model = resblock(next_in_model, num_channels)
    next_in_model = Conv2D(GRAYSCALE, (3, 3), padding='same')(next_in_model)
    next_in_model = Add()([initial, next_in_model])
    return Model(initial, next_in_model)


def train_model(model,images,corruption_func,batch_size,steps_per_epoch,num_epochs,num_valid_samples):
    """
    training the model
    :param model: the NN
    :param images: images to be trained on
    :param corruption_func: corruption fun to be applied on the images
    :param batch_size: size of batch per epoch
    :param steps_per_epoch: stepes per epoch
    :param num_epochs: amount of epochs as in how many times to run the NN
    :param num_valid_samples: size of validation set to compare to
    :return: a trained model
    """
    training_size = int(len(images)*EIGHTY_PERCENT_FACTOR)
    crop_size = model.input_shape[1:3]
    training = load_dataset(images[:training_size],batch_size,corruption_func,crop_size)
    validation = load_dataset(images[training_size:],batch_size,corruption_func,crop_size)
    model.compile(loss='mean_squared_error',optimizer=Adam(beta_2=0.9))
    model.fit_generator(training,steps_per_epoch,num_epochs,validation_data=validation,validation_steps=num_valid_samples) #what need to be numvalidsamples
    return model


def restore_image(corrupted_image, base_model):
    """
    restoring a corrupted image back to normal
    :param corrupted_image: corrupted image
    :param base_model: model to run on
    :return: restored image as type np.float 64 in range [0,1]
    """
    height, width = corrupted_image.shape
    shape = (height, width, 1)
    initial = corrupted_image.reshape(shape) - 0.5
    a = Input(shape=shape)
    b = base_model(a)
    new_model = Model(a, b)
    out = new_model.predict(initial[np.newaxis,...])
    ret = np.array(out[0,:,:,0]).astype(np.float64) + 0.5
    return ret.clip(0,1)


def help_round(im):
    """
    helper function, rounds each value i image to its closest i/255 and clips to range[0,1]
    :param im: image
    :return: rounded and clipped image
    """
    im = (im * MAX_PIX_LEVEL).astype(np.int) / MAX_PIX_LEVEL
    return im.clip(0,1)


def add_gaussian_noise(image,min_sigma,max_sigma):
    """
    corrupts image by adding noise spread out in a gaussian distribution
    :param image: image
    :param min_sigma: bottom range for sigma
    :param max_sigma: top range for sigma
    :return: corrupted image
    """
    sigma = np.random.uniform(min_sigma,max_sigma)
    corrupted = image + np.random.normal(ZERO_MEAN, sigma, image.shape)
    return help_round(corrupted)


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    """
    training a model to denoise
    :param num_res_blocks: amount of residual blocks
    :param quick_mode: if to run quick netwrok or full size
    :return: trained model
    """
    images = sol5_utils.images_for_denoising()
    corruption_func = lambda x: add_gaussian_noise(x,0,0.2)
    initial_model = build_nn_model(24,24,48,num_res_blocks)
    if quick_mode:
        trained_model = train_model(initial_model,images,corruption_func,10,3,2,30)
    else:
        trained_model = train_model(initial_model,images,corruption_func,100,100,5,1000)
    return trained_model


def add_motion_blur(image, kernel_size, angle):
    """
    adds motion blur, helper function for random_motion_blur
    :param image: image
    :param kernel_size: kernel size
    :param angle: angle of blur
    :return: blurred image
    """
    kernel = sol5_utils.motion_blur_kernel(kernel_size,angle)
    return convolve(image,kernel)


def random_motion_blur(image, list_of_kernel_sizes):
    """
    adds motion blur with random angle and kernel size
    :param image: image
    :param list_of_kernel_sizes: list of optional kernel sizes
    :return: blurred image
    """
    kernel_size = np.random.choice(list_of_kernel_sizes)
    angle = np.random.random() #gets number between 0 and 1
    angle = angle*np.pi
    corruption = add_motion_blur(image,kernel_size,angle)
    return help_round(corruption)


def learn_deblurring_model(num_res_blocks=5,quick_mode=False):
    """
    training a model to deblur
    :param num_res_blocks: amount of residual blocks
    :param quick_mode: if to run quick netwrok or full size
    :return: trained model
    """
    images = sol5_utils.images_for_deblurring()
    corruption_func = lambda x: random_motion_blur(x, [7])
    initial_model = build_nn_model(16,16,32,num_res_blocks)
    if quick_mode:
        trained_model = train_model(initial_model,images,corruption_func,10,3,2,30)
    else:
        trained_model = train_model(initial_model,images,corruption_func,100,100,10,1000)
    return trained_model


