from UNet import UNet
import dsbutils
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import numpy as np
from torch.utils.data.dataloader import default_collate
import dsbaugment
from torch.utils.data import DataLoader
from skimage import filters
from skimage.measure import label
from collections import OrderedDict
#region loss functions
# Aside:
# comparison between DICE and IOU: https://stats.stackexchange.com/questions/273537/f1-dice-score-vs-iou

#based on 'Generalised Dice overlap as a deep learning loss
#function for highly unbalanced segmentations' (Sudre et al, 2017)
def soft_dice_loss(inputs, targets, epsilon = 1):
    batch_size = targets.size(0)
    loss = 0.0
    m = nn.Sigmoid()
    for i in xrange(batch_size):
        prob = m(inputs[i])
        ref = targets[i]
        intersection_0 = ((1 - ref) * (1 - prob)).sum()
        union_0 = ((1 - ref) + (1 - prob)).sum()

        intersection_1 = (ref * prob).sum()
        union_1 = (ref + prob).sum()

        dl = 1 - (intersection_0+epsilon)/(union_0+epsilon) - (intersection_1+epsilon)/(union_1+epsilon)
        loss = loss + dl
    return loss / batch_size


def weighted_generalized_dice_loss(inputs, targets, weights=None):
    batch_size = targets.size(0)
    loss = 0.0

    for i in xrange(batch_size):
        prob = inputs[i]
        ref = targets[i]
        if weights is None:
            intersection_0 = ((1-ref) * (1-prob)).sum()
            union_0 = ((1-ref) + (1-prob)).sum()
        else:
            intersection_0 = (((1 - ref) * (1 - prob)) / weights[i]).sum()
            union_0 = ((1 - ref) + (1 - prob)).sum()
        freq_0 = (1-ref).sum()
        w0 = 1 / (freq_0 * freq_0)

        intersection_1 = (ref*prob).sum()
        union_1 = (ref + prob).sum()
        freq_1 = ref.sum()
        w1 = 1/(freq_1*freq_1)

        gdl = 1 - 2 * ((intersection_0*w0 + intersection_1*w1)/(w0*union_0+w1*union_1))
        loss = loss + gdl
    return loss/batch_size



# weighted cross entropy with optional border weights
def weighted_cross_entropy(inputs, targets, weights=None):
    batch_size = targets.size(0)
    loss = 0.0
    for i in xrange(batch_size):
        prob = inputs[i].view(-1)
        ref = targets[i].view(-1)
        w1 = prob.sum()
        w1 = (len(prob) - w1)/w1
        if weights is None:
            ce_loss = ((ref * prob.log()) + ((1-ref) * ((1-prob).log()))).mean()
        else:

            #ce_loss = (weights[i].view(-1)*(w1 * (ref * prob.log()) + ((1 - ref) * ((1 - prob).log())))).mean()
            ce_loss = ( (w1 * ref + weights[i].view(-1))*(ref * prob.log()) + ((1 - ref) * ((1 - prob).log()))).mean()

        loss = loss - ce_loss
    return loss/batch_size





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

def calc_expected_iou(labelled_mask):
    true_rles = dsbutils.get_rles_from_mask(labelled_mask, label_img=False)
    pred_rles = dsbutils.get_rles_from_mask(labelled_mask > 0, label_img=True)
    expected_iou = calc_avg_precision_iou(pred_rles, true_rles)
    return expected_iou

def try_add_weight_map(sample):
    '''

    when the Iou is low, it means that it's hard to label the binary mask, becuase cells are touching
    so we want the borders  to have more weight that all the other pixels
    :param sample:
    :param w0:
    :return:
    '''
    w0 = 1.0
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
    selected_keys = ['img', 'binary_mask']
    return collate_selected(batch, selected_keys)

def weighted_batch_collate(batch):
    selected_keys = ['img', 'binary_mask', 'weight_map']
    return collate_selected(batch, selected_keys)

def collate_selected(batch, selected_keys):
    selected_keys = [key for key in selected_keys if batch[0].get(key) is not None]
    return {key: default_collate([s.get(key) for s in batch]) for key in selected_keys}


# for train we do data augmentation as part of the transforms calls
def train(dataset, transformation, n_epochs, batch_size,
                               lr, weight_decay, momentum, weighted_loss,
                               init_weights, use_gpu, optimizer_type, verbose = True):

    # create the model
    unet = UNet(3,1, init_weights=init_weights)
    if use_gpu:
        unet.cuda()

    # define the loss criterion and the optimizer
    criterion = weighted_generalized_dice_loss
    if optimizer_type is None:
        optimizer_type = 'adam'

    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(unet.parameters(), lr=lr, momentum=momentum,
                                    weight_decay=weight_decay)
    else:
        # betas by default: beta1= 0.9, beta2=0.999
        optimizer = torch.optim.Adam(unet.parameters(), lr=lr, weight_decay=weight_decay)

    # assign the transformations
    dataset.transform = transformation
    if weighted_loss:
        train_loader = DataLoader(dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=8, collate_fn=weighted_batch_collate)
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=True, num_workers=8, collate_fn=batch_collate)

    #https: // arxiv.org / pdf / 1711.00489.pdf dont decay the learning rate, increase batch size
    # when the loss is 'stuck' - increase the batch size by some delta, up to a max size << total dataset size

    check_loss_change_freq = 3
    min_loss_change = 0.01
    batch_increase_delta = 2
    max_batch_size = 30
    prev_loss = None
    loss_change = 0.0
    reached_max_batch_size = False
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        mean_epoch_loss = 0.0
        for i, sample_batched in enumerate(train_loader):

            # get the inputs
            imgs = sample_batched.get('img')
            masks = sample_batched.get('binary_mask')

            if use_gpu:
                imgs, masks = Variable(imgs).cuda(), Variable(masks, requires_grad=False).cuda()
            else:
                imgs, masks = Variable(imgs), Variable(masks, requires_grad=False)
            weight_maps = None
            if weighted_loss:
                weight_maps = sample_batched.get('weight_map')
                if use_gpu:
                    weight_maps = Variable(weight_maps, requires_grad=False).cuda()
                else:
                    weight_maps = Variable(weight_maps, requires_grad=False)


            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = unet(imgs)

            loss = criterion(outputs, masks, weights=weight_maps)

            if not reached_max_batch_size:
                if prev_loss is None:
                    prev_loss = loss.data[0]
                    loss_change = prev_loss
                else:
                    loss_change = loss_change + prev_loss-loss.data[0]

            mean_epoch_loss = mean_epoch_loss + loss.data[0]
            loss.backward()
            optimizer.step()

        if verbose:
            print("epoch {} / {} : mean loss is: {}".format(epoch+1, n_epochs, mean_epoch_loss/(i+1)))
        if not reached_max_batch_size and (epoch+1)%check_loss_change_freq== 0:
            if loss_change/check_loss_change_freq < min_loss_change:

                if batch_size + batch_increase_delta > max_batch_size:
                    reached_max_batch_size = True
                    print("reached max batch size")
                else:
                    batch_size = batch_size + batch_increase_delta
                    batch_sampler = torch.utils.data.sampler.BatchSampler(train_loader.sampler, batch_size, False)
                    train_loader.batch_sampler = batch_sampler
                    print("increased batch size to {}".format(batch_size))
            prev_loss = loss.data[0]
            loss_change = 0.0
    return unet



def predict_masks(unet, dataset):
    raw_predicted_masks = OrderedDict()
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
        raw_predicted_masks[img_id] = dsbaugment.reverse_test_transform(predicted_mask, original_size) # predicted mask is now a numpy image again

    return raw_predicted_masks



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
    mean_avg_precision_iou = sum_avg_precision_iou / len(predictions.keys())
    return mean_avg_precision_iou

def test(models, dataset, requires_loading, postprocess, n_masks_to_collect=15):
        dataset.transform = dsbaugment.transformations.get("test_transform")
        raw_predicted_masks = OrderedDict()
        n_models = len(models)
        for model in models:
            if requires_loading:
                unet = torch.load(model)
            else:
                unet = model
            my_raw_predicted_masks = predict_masks(unet, dataset)
            for img_id, my_mask in my_raw_predicted_masks.items():
                mask = raw_predicted_masks.get(img_id)
                if mask is None:
                    raw_predicted_masks[img_id] = my_mask
                else:
                    raw_predicted_masks[img_id] = mask + my_mask

        i = 0
        predictions = {}
        examples = {}
        for img_id, raw_predicted_mask in raw_predicted_masks.items():

            raw_predicted_mask = raw_predicted_mask/n_models

            if postprocess:
                if len(np.unique(raw_predicted_mask)) > 1:

                    thresh = filters.threshold_otsu(raw_predicted_mask)
                    predicted_mask = (raw_predicted_mask > thresh).astype(np.uint8)
                    if np.sum(predicted_mask == 1) > np.sum(predicted_mask == 0):
                        predicted_mask = 1 - predicted_mask
                    predicted_mask = label(predicted_mask)

                    '''
                    import cv2
                    from skimage.color import gray2rgb, rgb2gray
                    #apply watershed from: https: // docs.opencv.org / 3.1.0 / d3 / db4 / tutorial_py_watershed.html
                    #ret, thresh = cv2.threshold((predicted_mask*255).astype(np.unit8), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    #print(thresh)
                    # noise removal
                    #predicted_mask = (raw_predicted_mask*255).astype(np.uint8)
                    #ret, predicted_mask = cv2.threshold(predicted_mask, 0, 255,
                    #                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    kernel = np.ones((6, 6), np.uint8)
                    opening = cv2.morphologyEx(predicted_mask, cv2.MORPH_OPEN, kernel, iterations=2)
                    # sure background area
                    sure_bg = cv2.dilate(opening, kernel, iterations=3)
                     # Finding sure foreground area
                    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
                    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 1, 0)
                    # Finding unknown region
                    sure_fg = np.uint8(sure_fg)
                    unknown = cv2.subtract(sure_bg, sure_fg)
                    # Marker labelling
                    ret, markers = cv2.connectedComponents(sure_fg)
                    # Add one to all labels so that sure background is not 0, but 1
                    markers = markers + 1
                    # Now, mark the region of unknown with zero
                    markers[unknown == 255] = 0
                    markers = cv2.watershed(gray2rgb(predicted_mask), markers)
                    markers[markers ==-1] = 0
                    predicted_mask = rgb2gray(markers)

                    '''
                else:
                    predicted_mask = label(raw_predicted_mask > 0.5)
            else:
                thresh = 0.5
                predicted_mask = label(raw_predicted_mask > thresh)


            pred_rles = dsbutils.get_rles_from_mask(predicted_mask)
            predictions[img_id] = pred_rles
            # save a few examples for plotting
            if i < n_masks_to_collect:
                sample = dataset[i]
                original_size = sample.get('size')
                img = sample.get('img')
                examples[img_id] = {'img': dsbaugment.reverse_test_transform(img, original_size),
                                    'raw_predicted_mask': raw_predicted_mask, 'predicted_mask': predicted_mask,
                                    "true_mask": sample.get('labelled_mask')  # can be Noe for the test set
                                    }
            i = i + 1
        return predictions, examples

#endregion
