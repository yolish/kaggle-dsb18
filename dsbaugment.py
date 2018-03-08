from PIL import Image
from torchvision import transforms
from JointCompose import JointCompose, IMG_ONLY_TRANSFORM, MASK_ONLY_TRANSFORM, RANDOM_JOINT_TRANSFORM_WITH_BORDERS, BORDER_ONLY_TRANSFORM, JOINT_TRANSFORM_WITH_BORDERS
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage.segmentation import mark_boundaries, find_boundaries
from skimage.exposure import adjust_gamma, rescale_intensity
import random



IMG_SIZE = 128


class TransformSpec:
    def __init__(self, transform, transform_type, prob = None):
        self.transform = transform
        self.transform_type = transform_type
        self.prob = prob


class Flip(object):
    """flips the given PIL Image horizontally or vertically.
    param type: 0 for horizontal flip, 1 for vertical flip
    """

    def __init__(self, flip_type):
        self.type = flip_type

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: flipped image.
        """
        if self.type == 0:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            return img.transpose(Image.FLIP_TOP_BOTTOM)




# aside: transforms are written as callable classes instead of simple functions so that parameters
#  of the transform need not be passed everytime it is called. For this, we just need to implement
# __call__ method and if required, __init__ method.
class JitterBrightness(object):
    def __call__(self, img):
        # img is a numpy rgb image
        gamma = random.random() + 0.3
        return adjust_gamma(img, gamma)

class Negative(object):
    def __call__(self, img):
        # img is a numpy rgb image
        return rescale_intensity(255-img)


class To3D(object):
    # make into a 3d RGB-like array required for making it a PIL image and then a tensor
    def __call__(self, mask):
        h = mask.shape[0]
        w = mask.shape[1]
        mask_rgb = np.zeros((h,w,3))
        for i in xrange(h):
            for j in xrange(w):
                if mask[i,j] == 1:
                    mask_rgb[i,j,:] = 255
        return mask_rgb.astype(np.uint8)

class To1Ch(object):
    def __call__(self, img, channel = 0):
        return img[:,:,channel][:,:,None]

class Binarize(object):
    def __call__(self, img):
        img[img > 0.5] = 1.0
        img[img < 1.0] = 0.0
        return img




class ElasticTransform(object):
    '''
    sigma: positive float for smoothing the transformation (elasticy of the transformation.)
    If sigma is small teh field looks like a completely random field after normalization
    For intermidiate sigma values the displacement fields look like elastic deformation, where sigma is the elasticity coefficient.
    If sigma is large, the displacements become close to affine. If sigma is very large the displacements become translations.
    alpha: scaling facor - positive float giving the intensity of the transformation. Larger alphas require larger sigmas
    default values take from the paper
    '''
    def __init__(self, sigma=2.0, alpha=17.0):
        '''

        :param sigma: positive floaf giving the elasticity of the transformation
        :param alpha: positive float giving the intensity of the transformation
        '''
        self.sigma = sigma
        self.alpha = alpha


    def __call__(self, img, mask, borders):

        if len(mask.shape) == 2:
            # merge the image and the mask
            merged_img = np.zeros(img.shape)
            merged_img[:,:,] = img[:,:,]
            merged_img[:,:,0] = mask[:,:]

            # apply elastic deformation on the merged image
            [deformed_merged_img, deformed_borders] = self.__elastic_deformation__([merged_img, borders])

            # split image and mask from the merged deformed image
            # mask
            deformed_mask = np.zeros(mask.shape)
            deformed_mask[:,:] = deformed_merged_img[:, :, 0]
            self.dichotom(deformed_mask, 0.5, 1.0)
            # image
            deformed_img = deformed_merged_img[:,:,:]
            deformed_img[:,:,0] = img[:,:,0]

        else:
            [deformed_img, deformed_mask, deformed_borders] = self.__elastic_deformation__([img, mask, borders])

        return deformed_img.astype(np.uint8), deformed_mask.astype(np.uint8), deformed_borders.astype(np.uint8)


    '''
    based on the paper 'Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis' Simard et al 2003
    generalized the following implementation: https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
    Works on numpy images
    '''
    def __elastic_deformation__(self, imgs):
        img = imgs[0]
        # img is a numpy image
        shape = img.shape
        n_dim = len(shape)
        convolved_displacement_fields = []
        grid = []
        fsize = len(img.flatten())
        for i in xrange(n_dim):
            if i < 2:  # don't touch the channel
                cdf = np.array([random.random() for j in xrange(fsize)]).reshape(shape) * 2 - 1
                convolved_displacement_fields.append(
                    gaussian_filter(cdf, self.sigma, mode="constant", cval=0) * self.alpha)
            grid.append(np.arange(shape[i]))
        grid = np.meshgrid(*grid, indexing='ij')
        indices = []
        for i in xrange(n_dim):
            if i < 2:  # don't touch the channel
                indices.append(np.reshape(grid[i] + convolved_displacement_fields[i], (-1, 1)))
            else:
                indices.append(np.reshape(grid[i], (-1, 1)))
        deformed_imgs = [map_coordinates(my_img, indices, order=3).reshape(shape) for my_img in imgs]
        return deformed_imgs

    def dichotom(self, img, thr, v1, v0=0):
        if len(img.shape) == 2:
            img[img > thr] = v1
            img[img < v1] = v0
        else:
            height, width, channel = img.shape
            for i in xrange(height):
                for j in xrange(width):
                    for k in xrange(channel):
                        if img[i, j, k] == thr:
                            img[i, j, :] = v1
                            break
            img[img < v1] = v0



def PIL_torch_to_numpy(img):
    img = np.transpose(img.numpy(), (1, 2, 0))
    if img.shape[2] == 1:
        img = img[:,:,0]
    return img

def reverse_test_transform(img, original_size):
    '''
    reverse the basic mask transformation
    :param img:
    :param original_size: H X W X C of image
    :return:
    '''
    # resize the tenstor to the original size
    reverse_transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize(original_size[:2]), transforms.ToTensor()])
    img = PIL_torch_to_numpy(reverse_transform(img))
    return img

def to_binary_mask(labelled_mask, with_borders, use_borders_as_mask):

    if use_borders_as_mask:
        mask = find_boundaries(labelled_mask, mode='outer')
    else:
        mask = (labelled_mask > 0)
        if with_borders:
            mask[find_boundaries(labelled_mask, mode='outer')] = 0

    borders = (labelled_mask > 0).astype(np.uint8) - mask  # borders of touching cells (if borders are marked)

    return mask.astype(np.uint8), borders.astype(np.uint8)


# add transformations to color

transformations = {
"train_transform_elastic":JointCompose(# transformations
    [
    # turn mask into 3D RGB-Like for PIL and tensor transformation
    TransformSpec(To3D(), MASK_ONLY_TRANSFORM),
    TransformSpec(To3D(), BORDER_ONLY_TRANSFORM),
    #Elastic deformation on the numpy images
    TransformSpec(ElasticTransform(), RANDOM_JOINT_TRANSFORM_WITH_BORDERS, prob=0.8),
    # Convert borders and mask to 1 channel
    TransformSpec(To1Ch(), BORDER_ONLY_TRANSFORM),
    TransformSpec(To1Ch(), MASK_ONLY_TRANSFORM),
    # color jittering (image only)
    TransformSpec(JitterBrightness(), IMG_ONLY_TRANSFORM, prob=0.9),
    TransformSpec(Negative(), IMG_ONLY_TRANSFORM, prob=0.5),
    # turn into a PIL image - required to apply torch transforms (both image, mask and borders)
    TransformSpec(transforms.ToPILImage(), JOINT_TRANSFORM_WITH_BORDERS),
    # flipping
    TransformSpec(Flip(1), JOINT_TRANSFORM_WITH_BORDERS, prob=0.5),

    #resize image (bilinear interpolation)
    TransformSpec(transforms.Resize((IMG_SIZE,IMG_SIZE),interpolation=Image.BILINEAR),
                  IMG_ONLY_TRANSFORM),
    #resize borders (bilinear interpolation)
    TransformSpec(transforms.Resize((IMG_SIZE,IMG_SIZE),interpolation=Image.NEAREST),
                  BORDER_ONLY_TRANSFORM),
    # resize mask
    TransformSpec(transforms.Resize((IMG_SIZE,IMG_SIZE),interpolation=Image.NEAREST),
                  MASK_ONLY_TRANSFORM),
    # finally turn into a torch tenstor (both image and mask)
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    TransformSpec(transforms.ToTensor(),JOINT_TRANSFORM_WITH_BORDERS),
    # ensure mask and borders are binarized
    TransformSpec(Binarize(), BORDER_ONLY_TRANSFORM),
    TransformSpec(Binarize(), MASK_ONLY_TRANSFORM)]
),
"train_transform":JointCompose(
    [TransformSpec(To3D(), MASK_ONLY_TRANSFORM),
    TransformSpec(To3D(), BORDER_ONLY_TRANSFORM),
    TransformSpec(To1Ch(), BORDER_ONLY_TRANSFORM),
    TransformSpec(To1Ch(), MASK_ONLY_TRANSFORM),
    TransformSpec(JitterBrightness(), IMG_ONLY_TRANSFORM, prob=0.5),
    TransformSpec(transforms.ToPILImage(), JOINT_TRANSFORM_WITH_BORDERS),
    TransformSpec(Flip(1), JOINT_TRANSFORM_WITH_BORDERS, prob=0.2),
    TransformSpec(transforms.Resize((IMG_SIZE,IMG_SIZE),interpolation=Image.BILINEAR),
                  IMG_ONLY_TRANSFORM),
    TransformSpec(transforms.Resize((IMG_SIZE,IMG_SIZE),interpolation=Image.NEAREST),
                  BORDER_ONLY_TRANSFORM),
    TransformSpec(transforms.Resize((IMG_SIZE,IMG_SIZE),interpolation=Image.NEAREST),
                  MASK_ONLY_TRANSFORM),
    TransformSpec(transforms.ToTensor(),JOINT_TRANSFORM_WITH_BORDERS),
     TransformSpec(Binarize(), BORDER_ONLY_TRANSFORM),
     TransformSpec(Binarize(), MASK_ONLY_TRANSFORM)]
),
"train_transform_jitter":JointCompose(
    [TransformSpec(To3D(), MASK_ONLY_TRANSFORM),
    TransformSpec(To3D(), BORDER_ONLY_TRANSFORM),
    TransformSpec(To1Ch(), BORDER_ONLY_TRANSFORM),
    TransformSpec(To1Ch(), MASK_ONLY_TRANSFORM),
    TransformSpec(JitterBrightness(), IMG_ONLY_TRANSFORM, prob=0.9),
    TransformSpec(Negative(), IMG_ONLY_TRANSFORM, prob=0.5),
    TransformSpec(transforms.ToPILImage(), JOINT_TRANSFORM_WITH_BORDERS),
    TransformSpec(Flip(1), JOINT_TRANSFORM_WITH_BORDERS, prob=0.2),
    TransformSpec(transforms.Resize((IMG_SIZE,IMG_SIZE),interpolation=Image.BILINEAR),
                  IMG_ONLY_TRANSFORM),
    TransformSpec(transforms.Resize((IMG_SIZE,IMG_SIZE),interpolation=Image.NEAREST),
                  BORDER_ONLY_TRANSFORM),
    TransformSpec(transforms.Resize((IMG_SIZE,IMG_SIZE),interpolation=Image.NEAREST),
                  MASK_ONLY_TRANSFORM),
    TransformSpec(transforms.ToTensor(),JOINT_TRANSFORM_WITH_BORDERS),
     TransformSpec(Binarize(), BORDER_ONLY_TRANSFORM),
     TransformSpec(Binarize(), MASK_ONLY_TRANSFORM)]
),
"test_transform":JointCompose(
    [TransformSpec(To3D(), MASK_ONLY_TRANSFORM),
TransformSpec(To3D(), BORDER_ONLY_TRANSFORM),
    TransformSpec(To1Ch(), BORDER_ONLY_TRANSFORM),
    TransformSpec(To1Ch(), MASK_ONLY_TRANSFORM),
    TransformSpec(transforms.ToPILImage(), JOINT_TRANSFORM_WITH_BORDERS),
    TransformSpec(transforms.Resize((IMG_SIZE,IMG_SIZE),interpolation=Image.BILINEAR),
                  IMG_ONLY_TRANSFORM),
    TransformSpec(transforms.Resize((IMG_SIZE,IMG_SIZE),interpolation=Image.NEAREST),
                  BORDER_ONLY_TRANSFORM),
    TransformSpec(transforms.Resize((IMG_SIZE,IMG_SIZE),interpolation=Image.NEAREST),
                  MASK_ONLY_TRANSFORM),
    TransformSpec(transforms.ToTensor(),JOINT_TRANSFORM_WITH_BORDERS),
     TransformSpec(Binarize(), BORDER_ONLY_TRANSFORM),
     TransformSpec(Binarize(), MASK_ONLY_TRANSFORM)]
),
"toy_transform":JointCompose(
    [TransformSpec(To3D(), MASK_ONLY_TRANSFORM),
TransformSpec(To3D(), BORDER_ONLY_TRANSFORM),
    TransformSpec(ElasticTransform(), RANDOM_JOINT_TRANSFORM_WITH_BORDERS, prob=1.0),

    TransformSpec(To1Ch(), BORDER_ONLY_TRANSFORM),
    TransformSpec(To1Ch(), MASK_ONLY_TRANSFORM),
    TransformSpec(transforms.ToPILImage(), JOINT_TRANSFORM_WITH_BORDERS),
    TransformSpec(Flip(1), JOINT_TRANSFORM_WITH_BORDERS, prob=0.0),

    TransformSpec(transforms.Resize((IMG_SIZE,IMG_SIZE),interpolation=Image.BILINEAR),
                  IMG_ONLY_TRANSFORM),
    TransformSpec(transforms.Resize((IMG_SIZE,IMG_SIZE),interpolation=Image.NEAREST),
                  BORDER_ONLY_TRANSFORM),
    TransformSpec(transforms.Resize((IMG_SIZE,IMG_SIZE),interpolation=Image.NEAREST),
                 MASK_ONLY_TRANSFORM),
    TransformSpec(transforms.ToTensor(),JOINT_TRANSFORM_WITH_BORDERS),
    TransformSpec(Binarize(), BORDER_ONLY_TRANSFORM),
    TransformSpec(Binarize(), MASK_ONLY_TRANSFORM)]
)
}






