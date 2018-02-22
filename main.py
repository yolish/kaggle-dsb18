from NucleiDataset import NucleiDataset

import dsbutils
import dsbml
import os
import torch
from UNet import UNet
import random
import numpy as np
import dsbaugment
#TODO architecture, run e2e
#TODO optimizer, weight init., update loss based on paper, update architecture based on paper
#TODO fix seeds
#TODO introduce loss with more weights for pixels in the boundaries: 1. generate masks with boundaries 2. generate weight maps
#TODO debug loss
#TODO cross validation;  sanity, save model between epochs?

if __name__ == "__main__":
    # general parameters
    dsb_data_path = "/home/yoli/kaggle_dsb18/data"
    dsb_output_path = "/home/yoli/kaggle_dsb18/predictions"
    stage = "stage1"
    sanity_basic = True
    sanity_augment = True

    visualize = True
    model_filename = None
    learn_and_predict = False

    # set seeds




    action = "collecting images details"
    time = dsbutils.start_action(action)
    imgs_details = dsbutils.collect_imgs_details(dsb_data_path, stage)
    dsbutils.complete_action(action, time)
    #print(imgs_details.sample(3))

    action = "creating train and validation datasets"
    validation_frac = 0.1
    time = dsbutils.start_action(action)
    labels_file = os.path.join(dsb_data_path, '{}_train_labels.csv'.format(stage))
    train_dataset = NucleiDataset('train', imgs_df=imgs_details, labels_file=labels_file)
    valid_dataset = train_dataset.split(validation_frac, 'validation')
    print("train size: {}, validation size: {}".format(len(train_dataset),
                                                        len(valid_dataset)))
    dsbutils.complete_action(action, time)

    if sanity_basic:
        print("performing a basic sanity check")
        if visualize:
            dsbutils.plot_imgs(train_dataset, 6, (17, 22))
        n_imgs = 3
        selected_idx = random.sample(range(len(train_dataset)), n_imgs)
        for img_idx in selected_idx:
            sample = train_dataset[img_idx]
            img_id = sample.get('id')
            img = sample.get('img')
            mask = sample.get('labelled_mask')
            rles_from_df = dsbutils.get_rles_from_df(train_dataset.dataset, img_id)
            rles_from_mask = dsbutils.get_rles_from_mask(mask, label_img=False)
            dsbutils.test_rle(rles_from_mask, rles_from_df)
            print("Avg precision IoU for img {} (using labelled): {}".format(img_id, dsbml.calc_avg_precision_iou(rles_from_mask,rles_from_df)))


    if sanity_augment:
        import matplotlib.pyplot as plt
        print("performing an augmentation sanity check")
        n_imgs = 1
        selected_idx = random.sample(range(len(train_dataset)), n_imgs)
        for img_idx in selected_idx:
            sample = train_dataset[img_idx]
            img_id = sample.get('id')
            img = sample.get('img')
            mask = sample.get('binary_mask')
            borders = sample.get('borders')
            print("image and binary mask before transformation. Image shape: {}, mask shape: {}".format(img.shape,mask.shape))
            plt.subplot(131)
            plt.imshow(img)
            plt.subplot(132)
            plt.imshow(mask, cmap='gist_gray')
            plt.subplot(133)
            plt.imshow(borders)
            plt.show()

        train_dataset.transform = dsbaugment.toy_transform
        for img_idx in selected_idx:
            sample = train_dataset[img_idx]
            img_id = sample.get('id')
            img = sample.get('img')
            mask = sample.get('binary_mask')
            borders = sample.get('borders')
            print("image and binary mask after transformation. Image shape: {}, mask shape: {}".format(img.shape,
                                                                                                        mask.shape))
            plt.subplot(131)
            plt.imshow(dsbaugment.PIL_torch_to_numpy(img))
            plt.subplot(132)
            plt.imshow(dsbaugment.PIL_torch_to_numpy(mask), cmap='gist_gray')
            plt.subplot(133)
            plt.imshow(dsbaugment.PIL_torch_to_numpy(borders), cmap='gist_gray')
            plt.show()



    if learn_and_predict:
        unet = None

        n_epochs = 2
        lr = 1e-04
        weight_decay = 2e-05
        momentum = 0.9
        batch_size = 4

        hyper_params = {'st':stage, 'lr':lr, 'wd':weight_decay,
                        'mm': momentum, 'bs':batch_size, 'epc':n_epochs}

        if model_filename is None:
            #TODO: cross validation on hyper-params
            action = "training a UNet"
            time = dsbutils.start_action(action)
            unet = dsbml.train(train_dataset, n_epochs= n_epochs, lr=lr,
                               weight_decay=weight_decay, batch_size=batch_size,
                               momentum=momentum)
            dsbutils.complete_action(action, time)

            action = "making predictions for the validation set"
            time = dsbutils.start_action(action)
            predictions, examples = dsbml.test(unet, valid_dataset)
            dsbutils.complete_action(action, time)

            action = "evaluating prediction"
            mean_avg_precision_iou = dsbml.evaluate(predictions, valid_dataset, examples=examples)
            print("IoU for validation dataset: {}".format(mean_avg_precision_iou))
            dsbutils.complete_action(action, time)
            if visualize:
                # visually evaluate a few images by comparing images and masks
                dsbutils.plot_predictions(examples, (7, 12))

            # train the model on the full train set (train + validation)
            action = "creating the full train dataset"
            time = dsbutils.start_action(action)
            train_dataset = NucleiDataset('train', imgs_df=imgs_details, labels_file=labels_file)
            print("train size: {}".format(len(train_dataset)))
            dsbutils.complete_action(action, time)

            action = "training a UNet on the full train dataset"
            time = dsbutils.start_action(action)
            unet = dsbml.train(train_dataset, n_epochs= n_epochs, lr=lr,
                               weight_decay=weight_decay, batch_size=batch_size,
                               momentum=momentum)
            # save the model
            model_filename = dsbutils.generate_filename(dsb_output_path, hyper_params, 'pth')
            torch.save(unet.state_dict(), model_filename)
            print("model written to to: {}".format(model_filename))

            dsbutils.complete_action(action, time)
        else:
            unet = UNet()
            unet = unet.load_state_dict(torch.load(model_filename))

        action = "creating the test set"
        time = dsbutils.start_action(action)
        test_dataset = NucleiDataset('test', imgs_df=imgs_details)
        print("test size: {}".format(len(test_dataset)))
        dsbutils.complete_action(action, time)

        action = "making predictions for the test set"
        time = dsbutils.start_action(action)
        predictions, examples = dsbml.test(unet, test_dataset)
        dsbutils.complete_action(action, time)
        # visually evaluate a few images by comparing images and masks
        dsbutils.plot_predictions(examples, (7, 12))

        action = "writing the predictions to submission format"
        time = dsbutils.start_action(action)
        submission_df = dsbutils.to_submission_df(predictions)
        #TODO: submission_filename - add hyper paroameters here or a time stamp?
        submission_filename = dsbutils.generate_filename(dsb_output_path, hyper_params, 'csv')
        submission_df.to_csv(submission_filename)
        print("predictions on tess set written to: {}".format(submission_filename))
        dsbutils.complete_action(action, time)


