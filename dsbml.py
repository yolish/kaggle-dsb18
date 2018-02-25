from UNet import UNet
import dsbutils
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.utils.data.dataloader import default_collate
import dsbaugment
import time
from torch.utils.data import DataLoader
from skimage import filters
import cv2

#region loss functions
# Aside:
# comparison between DICE and IOU: https://stats.stackexchange.com/questions/273537/f1-dice-score-vs-iou


# from https://www.kaggle.com/cloudfall/pytorch-tutorials-on-dsb2018
def soft_dice_loss(inputs, targets):
    num = targets.size(0)
    m1 = inputs.view(num, -1)
    m2 = targets.view(num, -1)
    intersection = (m1 * m2)
    dice = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1) # gives a score for each row
    loss = 1 - dice.sum() / num # 1- mean row score
    return loss

# approximation of IoU loss for binary (mask/background) output from 'Optimizing Intersection-Over-Union in Deep
# Neural Networks for Image Segmentation' Rhaman and Wang (http://www.cs.umanitoba.ca/~ywang/papers/isvc16.pdf)
'''
def iou_loss(inputs, targets):
    num = targets.size(0)
    m1 = inputs.view(num, -1)
    m2 = targets.view(num, -1)
    intersection = (m1 * m2).sum(1)
    union = (m1 + m2 - m1*m2).sum(1)
    iou = intersection/union  # gives a score for each row
    loss = 1 - iou.sum() / num  # 1- mean row score
    return loss
'''

# from https://github.com/pytorch/pytorch/issues/751
class StableBCELoss(nn.modules.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target, weight=None):
        neg_abs = - input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        if weight is not None:
            loss = loss * weight
        return loss.mean()

#endregion

#region evaluation metrics
# see alternative impl. on: https://www.kaggle.com/wcukierski/example-metric-implementation/notebook
def calc_avg_precision_iou(pred_rles, true_rles, thrs = np.arange(0.5, 1.0, 0.05)):
    '''
    # given true rles and predicted rles for a given image
    # decode them
    # calculate the intersection and union for each rle pair
    # for each threshold t, calculate TP, FP, FN:
    # A true positive is counted when a single predicted object matches a ground truth object with an IoU > t.
    # A false positive indicates a predicted object had no associated ground truth object.
    # A false negative indicates a ground truth object had no associated predicted object.
    # so all the pred RLE with at least one IoU > t are TP
    # all other pred RLEs are FP
    # all true RLEs with all IoU <= t are FN
    # return the mean IoU over all threshold
'''
    pred_rlds = [dsbutils.decode_rle(rle) for rle in pred_rles]
    true_rlds = [dsbutils.decode_rle(rle) for rle in true_rles]

    total_pred = len(pred_rlds)
    total_true = len(true_rlds)
    iou_scores = np.zeros(shape=(total_pred, total_true))

    for i in xrange(total_pred):
        for j in xrange(total_true):
            pred_rld = set(pred_rlds[i])
            true_rld = set(true_rlds[j])
            intersection = len(pred_rld.intersection(true_rld))
            union = len(pred_rld) + len(true_rld) - intersection
            iou_scores[i,j] = float(intersection)/union

    avg_precision_iou = 0.0
    for t in thrs:
        pred_with_match = np.sum(iou_scores > t, 1)
        true_with_match = np.sum(iou_scores > t, 0)
        pred_with_match[pred_with_match > 0] = 1
        true_with_match[true_with_match > 0] = 1
        tps = np.sum(pred_with_match)
        fps = total_pred - tps
        fns = total_true - np.sum(true_with_match)
        precision_iou = float(tps) / (tps + fps + fns)
        avg_precision_iou = avg_precision_iou + precision_iou
    return avg_precision_iou/len(thrs)

def calc_expected_iou(labelled_mask, binary_mask):
    true_rles = dsbutils.get_rles_from_mask(labelled_mask, label_img=False)
    pred_rles = dsbutils.get_rles_from_mask(binary_mask)
    expected_iou = calc_avg_precision_iou(pred_rles, true_rles)
    return expected_iou

def try_add_weight_map(sample, w0=4.0):
    '''

    when the Iou is low, it means that it's hard to label the binary mask, becuase cells are touching
    so we want the borders  to have more weight that all the other pixels
    :param sample:
    :param w0:
    :return:
    '''
    expected_iou = sample.get('expected_iou')
    borders = sample.get('borders')
    if expected_iou is not None and borders is not None:
        if isinstance(borders, np.ndarray):
            weight_map = borders.astype(np.float64)
        else:
            weight_map = borders.clone()
        weight_map[weight_map >= 1.0] = 1.0 + w0*(1-expected_iou)
        weight_map[weight_map < 1.0] = 1.0
        sample['weight_map'] = weight_map

#endregion


#regipp learning
# trap model with the train set
# thip consider: weight init., learning rate, optim, regularization,
# params of nets, batch size, loss, weight matrix for segmentation
# helping aids: pytorch tutorial + https://github.com/ycszen/pytorch-seg/blob/master/trainer.py/tester.py

# update based on this:

#Note: for train we do data augmentation as part of the transform

def batch_collate(batch):
    selected_keys = [key for key in ('img', 'binary_mask', 'expected_iou') if batch[0].get(key) is not None]
    return {key: default_collate([s.get(key) for s in batch]) for key in selected_keys}

# for train we do data augmentation as part of the transforms calls
def train(dataset, transformation, n_epochs, batch_size,
                               lr, weight_decay, momentum, weighted_loss, init_weights, use_gpu):
    # assign the transformations
    dataset.transform = transformation
    train_loader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=True, num_workers=8, collate_fn=batch_collate)

    # create the model
    unet = UNet(3,1, init_weights=init_weights)
    if use_gpu:
        unet.cuda()

    # define the loss criterion and the optimizer
    criterion = StableBCELoss()
    optimizer = torch.optim.SGD(unet.parameters(), lr=lr, momentum=momentum,
                                weight_decay=weight_decay)
    #optimizer = torch.optim.Adam(unet.parameters(), lr=lr)
    versbose_freq = 15
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        for i, sample_batched in enumerate(train_loader):

            # get the inputs
            start_time = time.time()
            imgs = sample_batched.get('img')
            masks = sample_batched.get('binary_mask')
            weight_maps = sample_batched.get('weight_map')
            if use_gpu:
                imgs, masks = Variable(imgs.cuda()), Variable(masks.cuda())
            else:
                imgs, masks = Variable(imgs), Variable(masks)
            if weighted_loss:
                if use_gpu:
                    weight_maps = Variable(weight_maps.cuda())
                else:
                    weight_maps = Variable(weight_maps)
            else:
                weight_maps = None

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = unet(imgs)
            loss = criterion(outputs, masks, weight=weight_maps )
            loss.backward()
            optimizer.step()


            # print statistics
            if i % versbose_freq == versbose_freq-1: # print every verbose_freq batches
                print("loss in epoch {} after processing {} images is {}".format(epoch + 1,
                                                                                 (i+1)*batch_size, loss.data[0]))

    return unet


def test(unet, dataset, postprocess=False, n_masks_to_collect=6):

    dataset.transform = dsbaugment.transformations.get("test_transform")
    i = 0
    examples = {}
    predictions = {}
    for sample in dataset:
        # apply the model to make predictions for the image
        img = default_collate([sample.get('img')])
        if next(unet.parameters()).is_cuda:
            img = Variable(img.cuda())
        else:
            img = Variable(img)
        predicted_mask = (unet(img)).data[0].cpu()
        # resize the mask and reverse the transformation
        img_id = sample.get('id')
        original_size = sample.get('size')
        # get the predicted mask (raw, i.e. peobabilities)
        raw_predicted_mask = dsbaugment.reverse_test_transform(predicted_mask, original_size) # predicted mask is now a numpy image again

        if postprocess:
            thresh = filters.threshold_otsu(raw_predicted_mask)
            predicted_mask = raw_predicted_mask > thresh
            # from: https://www.kaggle.com/gaborvecsei/basic-pure-computer-vision-segmentation-lb-0-229
            #mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            #mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        else:
            predicted_mask = raw_predicted_mask > 0.5

        pred_rles = dsbutils.get_rles_from_mask(predicted_mask)
        predictions[img_id] = pred_rles
        # save a few examples for plotting
        if i < n_masks_to_collect:
            examples[img_id] = {'img': dsbaugment.reverse_test_transform(img.data[0].cpu(), original_size),
                                'raw_predicted_mask':raw_predicted_mask, 'predicted_mask':predicted_mask,
                                "true_mask":sample.get('labelled_mask')# can be Noe for the test set
                                }
        i = i + 1
    return predictions, examples

def evaluate(predictions, dataset, examples=None):
    sum_avg_precision_iou = 0.0
    for img_id, pred_rles in predictions.items():
        true_rles = dsbutils.get_rles_from_df(dataset.dataset, img_id)
        avg_precision_iou = calc_avg_precision_iou(pred_rles, true_rles)
        sum_avg_precision_iou = sum_avg_precision_iou + avg_precision_iou
        if examples is not None:
            example = examples.get(img_id)
            if example is not None:
                example['iou'] = avg_precision_iou
    mean_avg_precision_iou = avg_precision_iou / len(dataset)
    return mean_avg_precision_iou

#endregion
