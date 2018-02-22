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
from PIL import Image

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

    def forward(self, input, target):
        neg_abs = - input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
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



#endregion

#region deep learning
# train the model with the train set
# things to consider: weight init., learning rate, optim, regularization,
# params of nets, batch size, loss, weight matrix for segmentation
# helping aids: pytorch tutorial + https://github.com/ycszen/pytorch-seg/blob/master/trainer.py/tester.py

# update based on this:

#Note: for train we do data augmentation as part of the transform

def batch_collate(batch):
    keys = ('img', 'binary_mask', 'borders')
    return {key: default_collate([d[key] for d in batch]) for key in batch[0] if key in keys}

# for train we do data augmentation as part of the transforms calls
def train(dataset, n_epochs, batch_size, lr, weight_decay, momentum):
    # assign the transformations
    dataset.transform = dsbaugment.train_transform
    train_loader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=True, num_workers=8, collate_fn=batch_collate)

    # create the model
    unet = UNet(3,1)

    # define the loss criterion and the optimizer
    #TODO: weight initialization
    #TODO: weight of boundary segmentation
    #TODO: different optimizer ?
    criterion = StableBCELoss()
    optimizer = torch.optim.SGD(unet.parameters(), lr=lr, momentum=momentum,
                                weight_decay=weight_decay)
    #optimizer = torch.optim.Adam(unet.parameters(), lr=lr)
    verbose_freq = 5
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, sample_batched in enumerate(train_loader):

            # get the inputs
            start_time = time.time()


            imgs = sample_batched.get('img')
            masks = sample_batched.get('binary_mask')
            #borders = sample_batched.get('borders')
            #weights = dsbaugment.compute_weight_map_from_borders(borders)

            # wrap them in Variable
            imgs, masks = Variable(imgs), Variable(masks)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize

            print("start forward")
            outputs = unet(imgs)
            print("end forward")

            loss = criterion(outputs, masks)
            print("loss is: {}".format(loss.data[0]))

            loss.backward()
            print("completed backward")
            optimizer.step()
            print("batch completed in {}".format(time.time()-start_time))
            # print statistics
            running_loss += loss.data[0]
            if i % verbose_freq == 0:
                print('running loss in epoch {}, batch{} is {}'.format(epoch + 1, i + 1, running_loss / verbose_freq))

            if k > -1:
                break
            k = k + 1
    return unet


# Note on data augmentation: for test we only do resizing and we do it inside the test loop
# i.e. not transforms added to the dataset
def test(unet, dataset, n_masks_to_collect=6, clean_mask=False):

    dataset.mask_transform = dsbaugment.test_transform

    i = 0
    examples = {}
    predictions = {}
    for sample in dataset:
        img = batch_collate([sample]).get('img')[0]
        img_id = sample.get('id')
        original_size = sample.get('size')
        img = Variable(img).unsqueeze(0)
        predicted_mask = unet(Variable(img))
        # resize the mask and reverse the transformation
        predicted_mask = dsbaugment.reverse_test_transform_for_mask(predicted_mask, original_size)
        # predicted mask is now a numpy image again

        # apply computer vision to clean the mask (disabled by default)
        if clean_mask:
            predicted_mask = dsbaugment.clean_mask(predicted_mask)

        pred_rles = dsbutils.rles_from_mask(predicted_mask)
        predictions[img_id] = pred_rles
        # save a few examples for plotting
        mask = sample.get('labelled_mask') # can be None for the test set
        if i < n_masks_to_collect:
            examples[img_id] = {img: img, 'labelled_mask': mask, 'predicted_mask': predicted_mask}
        i = i + 1
    return predictions, examples

def evaluate(predictions, dataset, examples=None):
    sum_avg_precision_iou = 0.0
    for img_id, pred_rles in predictions:
        true_rles = dsbutils.rles_from_df(dataset.dataset, img_id)
        avg_precision_iou = calc_avg_precision_iou(pred_rles, true_rles)
        sum_avg_precision_iou = sum_avg_precision_iou + avg_precision_iou
        if examples is not None:
            example = examples.get(img_id)
            if example is not None:
                example['iou'] = avg_precision_iou
    mean_avg_precision_iou = avg_precision_iou / len(dataset)
    return mean_avg_precision_iou

#endregion
