import numpy as np
import torch
import os
import time
from tqdm import tqdm

import sys
sys.path.append("../")

from utils.prep_utils import (
    COLORMAP,
    heatmaps_to_coordinates,
    N_KEYPOINTS,
    RAW_IMG_SIZE,
    MODEL_IMG_SIZE,
    show_batch_predictions,
    DATASET_MEANS,
    DATASET_STDS,
)

class Evaluator:

    @staticmethod
    def inference_fwd_baseline(model, test_dataloader):

        model = model.to('cpu')
        model.eval()
        accuracy_all = []
        inference_times = []

        for data in tqdm(test_dataloader):

            inputs = data["image"].to("cpu")
            true_keypoints = data["keypoints"].to("cpu").numpy()

            batch_size = inputs.shape[0]

            start_time = time.time()

            pred_heatmaps = model(inputs)
            pred_heatmaps = pred_heatmaps.detach().numpy()
            pred_keypoints = heatmaps_to_coordinates(pred_heatmaps)

            exec_time = time.time() - start_time

            inference_times.append(exec_time/batch_size)

            accuracy_keypoint = ((true_keypoints - pred_keypoints) ** 2).sum(axis=2) ** (1 / 2)
            accuracy_image = accuracy_keypoint.mean(axis=1)
            accuracy_all.extend(list(accuracy_image))

        error = np.mean(accuracy_all)
        exec_time_avg= np.mean(inference_times)
        print("Average error per keypoint: {:.1f}% from image size".format(error * 100))
        print("Execution time avg", exec_time_avg)

        MODEL_IMG_SIZE = 128
        RAW_IMG_SIZE = 224

        for img_size in [MODEL_IMG_SIZE, RAW_IMG_SIZE]:
            error_pixels = error * img_size
            image_size = f"{img_size}x{img_size}"
            print(
                "Average error per keypoint: {:.0f} pixels for image {}".format(
                    error_pixels, image_size
                )
            )

        return error, exec_time_avg
            
    @staticmethod
    def inference_fwd_resnet(model, test_dataloader):

        model = model.to('cpu')
        model.eval()
        accuracy_all = []
        inference_times = []

        for data in tqdm(test_dataloader):

            inputs = data["image"].to("cpu")
            true_keypoints = data["keypoints"].to("cpu").numpy()

            batch_size = inputs.shape[0]

            start_time = time.time()

            pred_keypoints = model(inputs)
            exec_time = time.time() - start_time
            pred_keypoints = pred_keypoints.detach().cpu().numpy()

            inference_times.append(exec_time/batch_size)

            true_keypoints = true_keypoints.reshape((batch_size, 21, 2))
            pred_keypoints = pred_keypoints.reshape((batch_size, 21, 2))

            accuracy_keypoint = ((true_keypoints - pred_keypoints) ** 2).sum(axis=2) ** (1 / 2)
            accuracy_image = accuracy_keypoint.mean(axis=1)
            accuracy_all.extend(list(accuracy_image))

        error = np.mean(accuracy_all)
        exec_time_avg= np.mean(inference_times)
        print("Average error per keypoint: {:.1f}% from image size".format(error * 100))
        print("Execution time avg", exec_time_avg)

        MODEL_IMG_SIZE = 224
        RAW_IMG_SIZE = 224

        for img_size in [MODEL_IMG_SIZE, RAW_IMG_SIZE]:
            error_pixels = error * img_size
            image_size = f"{img_size}x{img_size}"
            print(
                "Average error per keypoint: {:.0f} pixels for image {}".format(
                    error_pixels, image_size
                )
            )

        return error, exec_time_avg