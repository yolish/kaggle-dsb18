from NucleiDataset import NucleiDataset
import dsbutils
import dsbml
import os
import torch
import random
import dsbaugment
import json
import sys
import datetime
import numpy as np
import time

#TODO: e2e, training optimization, augmentation optimization, postprocessing (conditional random fields), documentation

if __name__ == "__main__":
    config_filename = sys.argv[1]

    with open(config_filename) as json_config_file:
        config = json.load(json_config_file)

    paths_config = config.get("paths")
    dsb_data_path = paths_config.get("input_path")
    dsb_output_path = paths_config.get("output_path")
    test_config = config.get("test")
    misc_config = config.get("misc")
    stage = misc_config.get("stage")
    postprocess = test_config.get("misc_config")


    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H-%M-%S")

    action = "collecting images details"
    start_time = dsbutils.start_action(action)
    imgs_details = dsbutils.collect_imgs_details(dsb_data_path, stage)
    dsbutils.complete_action(action, start_time)

    action = "creating the test set"
    start_time = dsbutils.start_action(action)
    test_dataset = NucleiDataset('test', imgs_df=imgs_details)
    print("test size: {}".format(len(test_dataset)))
    dsbutils.complete_action(action, start_time)


    models_filenames = []
    action = "making predictions for the test set with model ensemble"
    start_time = dsbutils.start_action(action)
    predictions, examples = dsbml.test_ensemble(models_filenames, test_dataset, postprocess=postprocess, n_masks_to_collect=0)
    dsbutils.complete_action(action, start_time)

    action = "writing the predictions to submission format"
    start_time = dsbutils.start_action(action)
    submission_df = dsbutils.to_submission_df(predictions)
    submission_filename = dsb_output_path + "ensemble_model_predictions_postprocess_" + str(
        postprocess) + "_" + timestamp + ".csv"
    submission_df.to_csv(submission_filename, columns=('ImageId', 'EncodedPixels'), index=False)
    print("predictions on tess set written to: {}".format(submission_filename))
    dsbutils.complete_action(action, start_time)

