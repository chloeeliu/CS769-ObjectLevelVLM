import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pycocotools.mask as mask
import yaml
from dataset.objectCOCO import COCOObjectDataset
from segment_anything import sam_model_registry, SamPredictor


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def main(args):
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    config['n_patches'] = 16
    if "336" in config['vision_encoder']:
        config['n_patches'] = 24

    dataset = COCOObjectDataset(config, split="val", n_patches=config['n_patches'],
                                max_examples_per_class=config["examples_per_class"])

    with open(args.original_outputs, 'rb') as file:
        outputs = pickle.load(file)
    with open(args.sam_outputs, 'rb') as file:
        outputs_sam = pickle.load(file)

    sam_improves = []

    for index, key in enumerate(outputs):
        if len(sam_improves) == args.k:
            break

        answer = outputs[key]["answer"]
        prediction = outputs[key]["prediction"]

        answer_sam = outputs_sam[key]["answer"]
        prediction_sam = outputs_sam[key]["prediction"]

        if (answer_sam.lower() in prediction_sam.lower()) and (answer.lower() not in prediction.lower()):
            sam_improves.append(dataset[index])

    sam = sam_model_registry[config["sam_model_type"]](checkpoint=config["sam_checkpoint"])
    predictor = SamPredictor(sam)

    for example in sam_improves:
        image = cv2.imread(example["path_to_image"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        original_mask = mask.decode(example["original_segmentation"])

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(original_mask, plt.gca())
        plt.savefig('analysis/' + example['id'] + '_original.png')
        plt.close()

        original_mask_points = np.stack(np.nonzero(original_mask)[::-1], -1)
        original_mask_mean = np.mean(original_mask_points, axis=0)

        predictor.set_image(image)
        mask_sam, _, _ = predictor.predict(
            point_coords=np.array([original_mask_mean]),
            point_labels=np.array([1]),
            multimask_output=False
        )

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask_sam[0], plt.gca())
        plt.savefig('analysis/' + example['id'] + '_sam.png')
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--original_outputs", type=str)
    parser.add_argument("--sam_outputs", type=str)
    parser.add_argument("--k", type=int)
    args = parser.parse_args()

    main(args)
