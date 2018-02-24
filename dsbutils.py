import numpy as np
import pandas as pd
from skimage.measure import label
import os
from glob import glob
import matplotlib.pyplot as plt
import matplotlib as mpl

import time
import random


# region dataset preparation
def read_imgs_labels(labels_file):
    # based on: https://www.kaggle.com/kmader/nuclei-overview-to-submission, with modifications
    imgs_labels = pd.read_csv(labels_file)
    imgs_labels['EncodedPixels'] = imgs_labels['EncodedPixels'].map(lambda ep: [int(x) for x in ep.split(' ')])
    return imgs_labels

def collect_imgs_details(data_dir, stage):
    # based on: https://www.kaggle.com/kmader/nuclei-overview-to-submission, with modifications
    all_imgs_paths = glob(os.path.join(data_dir, stage+'_*', '*', '*', '*'))
    imgs_df = pd.DataFrame({'path': all_imgs_paths})
    img_id = lambda in_path: in_path.split('/')[-3]
    img_type = lambda in_path: in_path.split('/')[-2]
    img_group = lambda in_path: in_path.split('/')[-4].split('_')[1]
    img_stage = lambda in_path: in_path.split('/')[-4].split('_')[0]
    imgs_df['ImageId'] = imgs_df['path'].map(img_id)
    imgs_df['ImageType'] = imgs_df['path'].map(img_type)
    imgs_df['TrainingSplit'] = imgs_df['path'].map(img_group)
    imgs_df['Stage'] = imgs_df['path'].map(img_stage)
    return imgs_df

def collect_dataset(imgs_df, dataset_type, labels_file = None):
    labels_df = None
    if labels_file is not None:
        labels_df = read_imgs_labels(labels_file)
    query = 'TrainingSplit==\"' + dataset_type + '\"'
    dataset_details = imgs_df.query(query)
    dataset_rows = []
    group_cols = ['Stage', 'ImageId']
    for n_group, n_rows in dataset_details.groupby(group_cols):
        img_record = {col_name: col_value for col_name, col_value in zip(group_cols, n_group)}
        img_paths = n_rows.query('ImageType == "images"')['path'].values.tolist()
        assert(len(img_paths) == 1)
        img_record['ImagePath'] = n_rows.query('ImageType == "images"')['path'].values.tolist()[0]
        if labels_df is not None:
            img_record['MaskPaths'] = n_rows.query('ImageType == "masks"')['path'].values.tolist()
            img_record['EncodedPixels'] = labels_df[labels_df['ImageId'] == img_record.get('ImageId')]['EncodedPixels'].values.tolist()
        img_record['Shape'] = None
        img_record['Channels'] = None
        dataset_rows += [img_record]
    dataset = pd.DataFrame(dataset_rows)
    return dataset
# endregion


#region visualization
def plot_imgs(dataset, n_imgs, fig_size, plot_mask=True):
    '''
    Plot the first n_imgs in the dataset with their masks (optional)
    :param dataset: a NucleiDataset instance
    :param n_imgs: the number of images to plot
    :param fig_size: the size of the figure to plot
    :param plot_mask: whether to plot the mask (True by default), applied only if a mask is available
    '''
    n_cols = 5
    if not plot_mask:
        n_cols = 1
    fig, axes = plt.subplots(n_imgs, n_cols, figsize=fig_size)
    norm = mpl.colors.Normalize(vmin = 1.0, vmax = 5.0)
    selected_idx = random.sample(range(len(dataset)), n_imgs)
    for i, img_idx in enumerate(selected_idx):
        sample = dataset[img_idx]
        subplot = axes[i][0]
        subplot.imshow(sample.get('img'))
        subplot.axis('off')
        if i == 0:
            subplot.set_title('Image')
        if plot_mask:
            subplot = axes[i][1]
            subplot.imshow(sample.get('labelled_mask'), cmap='magma')
            subplot.axis('off')
            if i == 0:
                subplot.set_title('Labelled Mask')
            subplot = axes[i][2]
            subplot.imshow(sample.get('binary_mask'), cmap='gist_gray')
            subplot.axis('off')
            if i == 0:
                subplot.set_title('Binary Mask')
            subplot = axes[i][3]
            subplot.imshow(sample.get('borders'))
            subplot.axis('off')
            if i == 0:
                subplot.set_title('Borders')

            subplot = axes[i][4]
            subplot.imshow(sample.get('weight_map')[:,:,0], cmap='magma', norm=norm)
            subplot.axis('off')
            if i == 0:
                subplot.set_title('Weight Map')
    plt.show()

def plot_predicted_masks(samples, fig_size, plot_true_mask=True):
    n_cols = 3
    n_imgs = len(samples)
    if not plot_true_mask:
        n_cols = 2
    fig, axes = plt.subplots(n_imgs, n_cols, figsize=fig_size)
    for i, sample in enumerate(samples.values()):
        img = sample.get('img')
        predicted_mask = sample.get('predicted_mask')
        mask = sample.get('labelled_mask')
        #iou = sample.get('iou') use

        subplot = axes[i][0]
        subplot.imshow(img)
        subplot.axis('off')
        if i == 0:
            subplot.set_title('Input')

        subplot = axes[i][1]
        subplot.imshow(predicted_mask, cmap='gist_gray')
        subplot.axis('off')
        if i == 0:
            subplot.set_title('Prediction')

        if plot_true_mask and mask is not None:
            subplot = axes[i][2]
            subplot.imshow(mask, cmap='magma')
            subplot.axis('off')
            if i == 0:
                subplot.set_title('True mask')


    plt.show()

#endregion

#region segmentation encoding/decoding (RLE)
def calc_rle(arr):
    # original code from: https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
    # minor changes (comments, renaming, etc.) done
    '''
    run-length encoding (rle) assumes that pixels are one-indexed and numbered from top to bottom, then left to right
    so, for example 1 is pixel (1,1), 2 is pixel (2,1), etc.
    full definition: https://www.kaggle.com/c/data-science-bowl-2018#evaluation
    the input is a labelled blob
    :param arr: numpy array of shape (height, width), 1 - mask, 0 - background
    :return: run length encoding for the given array
    '''
    # take the transpose to get down-then-right ordering, then flatten
    mask = np.where(arr.T.flatten()==1)[0]
    rle = []
    prev = -2
    for index in mask:
        if index > prev+1: # end of "stretch" - need to start a new run length
            rle.extend((index+1, 0)) # add the first pixel (1-indexed) and the run-length
        rle[-1] += 1 # increase the run-length by one until we get to the next "stretch"
        prev = index
    return rle

def decode_rle(rle):
    # decode a given run length encoding into a single array
    rld = []
    i = 0
    while i < len(rle):
        start_index = rle[i]
        length = rle[i+1]
        rld.extend(np.arange(start_index, start_index+length))
        i = i + 2
    return rld

def get_rles_from_df(imgs_df, img_id):
    rles = (imgs_df[imgs_df['ImageId'] == img_id]['EncodedPixels']).values[0]
    return sorted(rles, key = lambda x: x[0])


'''
def compute_weight_map(labelled_mask):
    shape = labelled_mask.shape
    weight_map = np.zeros(shape)
    n_labels = np.max(labelled_mask) + 1
    for label in xrange(1, n_labels+1):
        
    for i in xrange(0, shape[0]):
        for j in xrange(0, shape[1]):
            my_label = labelled_mask[i,j]
            temp = labelled_mask[labelled_mask != my_label]
'''



def get_rles_from_mask(labelled_img, thresh = 0.5, label_img = True):
    '''
    :param labelled_img: the image with the segmented objects; np array of shape (hight_, width)
           if the image is not labelled p > threshold - mask, p <= threshold - background
    :param thresh: the threshold to apply to convert class probabilities to mask vs background
    :param label_img: a boolean indicating whether to label the mask to get the different objects
    :return: a list of RLEs for the segmented objects in the given image
    '''
    rles = []
    if label_img:
        labelled_img = label(labelled_img > thresh) # Label connected regions of an integer array.

    n_labels = labelled_img.max()
    if n_labels<1:
        labelled_img[0,0] = 1 # ensure at least one mask per image
    for l in range(1, n_labels+1): # for each mask calculate its run-length encoding
        mask = labelled_img == l
        rle = calc_rle(mask)
        rles.append(rle)
    return sorted(rles, key = lambda x: x[0])
#endregion





#region post-processing for submission #
def format_rle(rle):
    return " ".join([str(i) for i in rle])

def to_submission_df(predictions):
    df = pd.DataFrame()
    for img_id, pred_rles in predictions:
        for rle in pred_rles:
            s = pd.Series({'ImageId': img_id, 'EncodedPixels': format_rle(rle)})
            df = df.append(s, ignore_index=True)
    return df
#endregion

#region testing and logging
# check rle function
# from: https://www.kaggle.com/kmader/nuclei-overview-to-submission with modifications
def test_rle(rles_from_mask, rles_from_df):
    match, mismatch = 0, 0
    # assume they are both sorted
    for mask_rle, df_rle in zip(rles_from_mask,rles_from_df):
        for i_x, i_y in zip(mask_rle, df_rle):
            if i_x == i_y:
                match += 1
            else:
                mismatch += 1
    print('Matches: %d, Mismatches: %d'% (match, mismatch))

def start_action(msg):
    start_time = time.time()
    print("start " + msg)
    return start_time

def complete_action(msg, start_time):
    print("completed " + msg)
    print("elapsed time: {}".format(time.time()-start_time))

# check mean IoU vs visualization
# test iou - make a fake prediction
#endregion
