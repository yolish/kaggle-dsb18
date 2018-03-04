import os
import random
import json
import sys
import datetime
import numpy as np
import time


#TODO: test weighted gdl. documentation (code doc, delete files and old predictions, etc),
#     run with large number of epochs over all train and do ensemble (bash?), figure out random ?

if __name__ == "__main__":
    config_filename = sys.argv[1]
    with open(config_filename) as json_config_file:
        config = json.load(json_config_file)

    paths_config = config.get("paths")
    dsb_data_path = paths_config.get("input_path")
    dsb_output_path = paths_config.get("output_path")

    actions_config = config.get("actions")
    sanity_basic = actions_config.get("sanity_basic")
    sanity_augment = actions_config.get("sanity_augment")
    visualize = actions_config.get("visualize")
    seed = actions_config.get('seed')
    validation_dataset_filename = actions_config.get('validation_dataset_filename')
    use_borders_as_mask = actions_config.get('use_borders_as_mask')
    if use_borders_as_mask is None:
        use_borders_as_mask = False
    add_borders_to_mask = actions_config.get('add_borders')
    if add_borders_to_mask is None:
        add_borders_to_mask = True


    train_config = config.get("train")
    test_config = config.get("test")
    misc_config = config.get("misc")
    stage = misc_config.get("stage")

    if seed is not None:
        # seed all random instances
        np.random.seed(seed)
        random.seed(seed)
    import torch
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    from NucleiDataset import NucleiDataset
    import dsbutils
    import dsbml
    import dsbaugment

    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H-%M-%S")

    action = "collecting images details"
    start_time = dsbutils.start_action(action)
    imgs_details = dsbutils.collect_imgs_details(dsb_data_path, stage)
    dsbutils.complete_action(action, start_time)
    #print(imgs_details.sample(3))

    action = "creating train and validation datasets"
    validation_frac = 0.1
    start_time = dsbutils.start_action(action)
    labels_file = os.path.join(dsb_data_path, '{}_train_labels.csv'.format(stage))
    train_dataset = NucleiDataset('train', imgs_df=imgs_details, labels_file=labels_file,
                                  add_borders_to_mask=add_borders_to_mask, use_borders_as_mask=use_borders_as_mask)
    if validation_dataset_filename is not None:
        valid_dataset = train_dataset.split(validation_frac, 'validation', filename= validation_dataset_filename)
    else:
        valid_dataset = train_dataset.split(validation_frac, 'validation')
        valid_dataset.dataset.to_csv(dsb_output_path + "validation_dataset_" + timestamp + ".csv")

    print("train size: {}, validation size: {}".format(len(train_dataset),
                                                        len(valid_dataset)))

    dsbutils.complete_action(action, start_time)

    if sanity_basic:
        print("performing a basic sanity check")
        dsbutils.plot_imgs(train_dataset, 3, (22, 27))
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
            print("Expected Avg precision IoU for img {}: {}".format(img_id, sample.get('expected_iou')))

    if sanity_augment:
        import matplotlib.pyplot as plt
        print("performing an augmentation sanity check")
        n_imgs = 2
        selected_idx = random.sample(range(len(train_dataset)), n_imgs)
        for img_idx in selected_idx:
            train_dataset.transform = None
            sample = train_dataset[img_idx]
            img_id = sample.get('id')
            img = sample.get('img')
            mask = sample.get('binary_mask')
            borders = sample.get('borders')
            weight_map = sample.get('weight_map')
            print("image and binary mask before transformation. Image shape: {}, mask shape: {}".format(img.shape,mask.shape))
            plt.subplot(231)
            plt.imshow(img)
            plt.subplot(232)
            plt.imshow(mask, cmap='gist_gray')
            plt.subplot(233)
            plt.imshow(borders, cmap = 'gist_gray')

            train_dataset.transform = dsbaugment.transformations.get("toy_transform")
            sample = train_dataset[img_idx]
            img_id = sample.get('id')
            img = sample.get('img')
            mask = sample.get('binary_mask')
            borders = sample.get('borders')
            print("image and binary mask after transformation. Image shape: {}, mask shape: {}".format(img.shape,
                                                                                                        mask.shape))
            plt.subplot(234)
            plt.imshow(dsbaugment.PIL_torch_to_numpy(img))
            plt.subplot(235)
            plt.imshow(dsbaugment.PIL_torch_to_numpy(mask), cmap='gist_gray')
            plt.subplot(236)
            plt.imshow(dsbaugment.PIL_torch_to_numpy(borders), cmap='gist_gray')
            plt.show()


    evaluate = False
    test = False
    postprocess = False
    unet = None
    requires_loading = False
    borders_model_filename = None
    if test_config is not None:
        evaluate = test_config.get("eval")
        test = test_config.get("test")
        postprocess = test_config.get("postprocess")
        model_filename = test_config.get("model")
        ensemble = test_config.get("ensemble")
        borders_model_filename = test_config.get("borders_model_filename")

        if ensemble is not None:
            requires_loading = True


    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H-%M-%S")

    if train_config is not None:
        model_filename = train_config.get("model")


        if model_filename is None:
            batch_size = train_config.get("batch_size")
            n_epochs = train_config.get("n_epochs")
            lr = train_config.get("lr")
            weight_decay = train_config.get("weight_decay")
            momentum = train_config.get("momentum")
            init_weights = train_config.get("init_weights")
            weighted_loss = train_config.get("weighted_loss")
            use_gpu = train_config.get("use_gpu")
            train_full = train_config.get("train_full")
            save_model = train_config.get("save_model")
            transformation_name = train_config.get("transformation")
            transformation = dsbaugment.transformations.get(transformation_name)
            optimizer = train_config.get('optimizer')
            hyperparam_search_config = train_config.get("hyperparam_search_config")

            if hyperparam_search_config:
                # do search for lr and weight_decay (regularization)
                # we do a random search rather than a grid search as it is more likely to catch the 'good points'
                # see Begstra and Bengayo 2012
                lr_log_range = hyperparam_search_config.get("lr_log_range")
                weight_decay_log_range = hyperparam_search_config.get("weight_decay_log_range")
                n_search = hyperparam_search_config.get("n_search")
                best_iou = 0.0
                best_params = {}
                for i in xrange(n_search):
                    action = "search {}".format(i+1)
                    start_time = dsbutils.start_action(action)
                    lr = 10**np.random.uniform(lr_log_range[0],lr_log_range[1])
                    weight_decay = 10**np.random.uniform(weight_decay_log_range[0], weight_decay_log_range[1])
                    unet = dsbml.train(train_dataset, transformation, n_epochs, batch_size,
                                       lr, weight_decay, momentum, weighted_loss, init_weights, use_gpu,
                                       optimizer)
                    predictions, _ = dsbml.test([unet], valid_dataset, requires_loading, postprocess, n_masks_to_collect=0)
                    mean_avg_precision_iou = dsbml.evaluate(predictions, valid_dataset)
                    if mean_avg_precision_iou > best_iou:
                        best_iou = mean_avg_precision_iou
                        best_params["search"] = i+1
                        best_params["lr"] = lr
                        best_params["weight_decay"] = weight_decay
                        best_params["iou"] = mean_avg_precision_iou
                    print("{}/{} lr: {} weight decay: {} IoU: {}".format(i+1, n_search, lr,
                                                                         weight_decay, mean_avg_precision_iou))
                    dsbutils.complete_action(action, start_time)
                print("completed hyper-parameter search, best performance found on search {}: lr: {} weight decay:{} IoU: {}".format(
                    best_params.get("search"), best_params.get("lr"), best_params.get("weight_decay"), best_params.get("iou")
                    # results following coarse and fine searcg:
                    #  lr: 0.00013
                    # weight decay: 3.47111926934e-07 (0.00000035)
                ))

            elif train_full:
                # train the model on the full train set (train + validation)
                action = "creating the full train dataset"
                start_time = dsbutils.start_action(action)
                train_dataset = NucleiDataset('train', imgs_df=imgs_details, labels_file=labels_file,
                                              add_borders_to_mask=add_borders_to_mask, use_borders_as_mask=use_borders_as_mask)
                print("train size: {}".format(len(train_dataset)))
                dsbutils.complete_action(action, start_time)

                action = "training a UNet on the full train dataset"
                start_time = dsbutils.start_action(action)
                unet = dsbml.train(train_dataset, transformation, n_epochs, batch_size,
                                   lr, weight_decay, momentum, weighted_loss, init_weights, use_gpu, optimizer)
                dsbutils.complete_action(action, start_time)

            else: # train without validation set
                action = "training a UNet"
                start_time = dsbutils.start_action(action)
                unet = dsbml.train(train_dataset, transformation, n_epochs, batch_size,
                                   lr, weight_decay, momentum, weighted_loss, init_weights, use_gpu,
                                   optimizer)
                dsbutils.complete_action(action, start_time)


            if save_model:
                model_filename = dsb_output_path + "model_" + timestamp + ".pth"
                torch.save(unet, model_filename)
                print("model written to to: {}".format(model_filename))
                model_metatdata_filename = dsb_output_path + "model_config_" + timestamp + ".txt"
                with open(model_metatdata_filename, 'w') as f:
                    f.write(json.dumps({"model_config":train_config}))
                print("model metadata written to to: {}".format(model_metatdata_filename))



    if evaluate or test:
        if unet is None:
            if model_filename is not None:
                unet = [model_filename]
                # take the timestamp from the model name
                timestamp = model_filename.split("model_")[1].split(".pth")[0]
                requires_loading = True
            else:
                unet = ensemble
        else:
            unet = [unet]

    if evaluate:
        action = "making predictions for the validation set"
        start_time = dsbutils.start_action(action)
        predictions, examples = dsbml.test(unet, valid_dataset, requires_loading, postprocess,
                                           borders_model_filename=borders_model_filename)
        dsbutils.complete_action(action, start_time)

        action = "evaluating predictions"
        start_time = dsbutils.start_action(action)
        mean_avg_precision_iou = dsbml.evaluate(predictions, valid_dataset, examples=examples)
        print("IoU for validation dataset: {}".format(mean_avg_precision_iou))
        dsbutils.complete_action(action, start_time)
        if visualize:
            # visually evaluate a few images by comparing images and masks
            dsbutils.plot_predicted_masks(examples, (22, 27))

    if test:
        action = "creating the test set"
        start_time = dsbutils.start_action(action)
        test_dataset = NucleiDataset('test', imgs_df=imgs_details, add_borders_to_mask=add_borders_to_mask)
        print("test size: {}".format(len(test_dataset)))
        dsbutils.complete_action(action, start_time)

        action = "making predictions for the test set"
        start_time = dsbutils.start_action(action)
        predictions, examples = dsbml.test(unet, test_dataset, requires_loading, postprocess,
                                           n_masks_to_collect=20, borders_model_filename=borders_model_filename)
        dsbutils.complete_action(action, start_time)
        # visually evaluate a few images by comparing images and masks
        if visualize:
            dsbutils.plot_predicted_masks(examples, (7, 12), plot_true_mask=False)

        action = "writing the predictions to submission format"
        start_time = dsbutils.start_action(action)
        submission_df = dsbutils.to_submission_df(predictions)
        if ensemble is not None:
           ensemble_metatdata_filename = dsb_output_path + "ensemble_config_" + timestamp + ".txt"
           with open(ensemble_metatdata_filename, 'w') as f:
               f.write(json.dumps({"ensemble_config": test_config}))
           timestamp = "ensemble_" + timestamp
        submission_filename = dsb_output_path + "model_predictions_postprocess_" + str(postprocess) + "_" + timestamp + ".csv"
        submission_df.to_csv(submission_filename, columns=('ImageId','EncodedPixels'), index=False)
        print("predictions on tess set written to: {}".format(submission_filename))
        dsbutils.complete_action(action, start_time)


