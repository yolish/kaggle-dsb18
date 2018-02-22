import numpy as np
IMG_ONLY_TRANSFORM = 1
MASK_ONLY_TRANSFORM = 2
JOINT_TRANSFORM = 3
RANDOM_JOINT_TRANSFORM_WITH_BORDERS = 4 # joint with randomness inside the transform that affect borders
BORDER_ONLY_TRANSFORM = 5
JOINT_TRANSFORM_WITH_BORDERS = 6

# ad hoc transform classes from https://github.com/ycszen/pytorch-seg/blob/master/transform.py

# joint transformations for image and mask
class JointCompose(object):
    def __init__(self, transforms_specs):
        self.transforms_specs = transforms_specs

    def __call__(self, sample):
        img = sample.get('img')
        mask = sample.get('binary_mask')
        borders = sample.get('borders')
        for transform_spec in self.transforms_specs:
            transform = transform_spec.transform
            transform_type = transform_spec.transform_type
            prob = transform_spec.prob

            # check if to apply the transform, in case of a probabilistic one
            apply_transform = True
            if prob is not None: # probabilistic transform
                if np.random.random() > prob:
                    apply_transform = False

            if apply_transform:
                if transform_type == IMG_ONLY_TRANSFORM:
                    img = transform(img)
                elif transform_type == JOINT_TRANSFORM:
                    img = transform(img)
                    if mask is not None:
                        mask = transform(mask)
                elif transform_type == JOINT_TRANSFORM_WITH_BORDERS:
                    img = transform(img)
                    if mask is not None:
                        mask = transform(mask)
                        borders = transform(borders)
                if mask is not None:
                    if transform_type == MASK_ONLY_TRANSFORM:
                        mask = transform(mask)
                    elif transform_type == RANDOM_JOINT_TRANSFORM_WITH_BORDERS:
                        img, mask, borders = transform(img, mask, borders)
                    elif transform_type == BORDER_ONLY_TRANSFORM:
                        borders = transform(borders)



        sample['img'] = img
        sample['binary_mask'] = mask
        sample['borders'] = borders
        return sample



