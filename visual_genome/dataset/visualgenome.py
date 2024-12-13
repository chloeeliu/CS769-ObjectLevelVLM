import json
import os
import numpy as np
import pycocotools.mask as mask
import math
import torch
import skimage
import pickle
import datasets

from tqdm import tqdm
from torch.utils.data import Dataset

import random
random.seed(42) 

class VisualGenomeDataset(Dataset):
    # Path to instances.json
    # Path to COCO2014/COCO2017 train images
    def __init__(self, config, split="train", patch_size=24, max_examples_per_class = 1000, images_num=None, entries_num=None):
        super(VisualGenomeDataset, self).__init__()
        self.images_num = images_num
        self.entries_num = entries_num
        self.config = config
        datasets.config.DOWNLOADED_DATASETS_PATH = "/u/h/k/hkhader/research/datasets/olive"
        self.dataset = datasets.load_dataset("visual_genome", "relationships_v1.2.0")
        self.patch_size = patch_size
        self.max_examples_per_class = max_examples_per_class
        self.class_counts = {}
        
        from datasets import load_dataset
        train_dataset, val_dataset = self.dataset['train'].train_test_split(test_size=0.2, seed=42).values()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        if split == "train":
            self.entries = self._load_dataset(train_dataset)
            if entries_num is not None:
                self.entries =  random.sample(self.entries, entries_num)
        elif split == "val":
            self.entries = self._load_dataset(val_dataset)
            if entries_num is not None:
                self.entries =  random.sample(self.entries, entries_num)
        else:
            raise Exception("Invalid split")

        if config["use_retrieval"]:
            retrieval_path = config["retrieval_set_path"]
            with open(retrieval_path, 'rb') as f:
                self.retrieval_data = pickle.load(f)
                self.retrieval_keys = torch.FloatTensor(self.retrieval_data['keys']).to(self.config["device"])
                self.retrieval_labels = self.retrieval_data['values']
                self.retrieval_idx = self.retrieval_data.get('idx', None)
                assert len(self.retrieval_keys) == len(self.retrieval_labels)
                print(f'Loaded {len(self.retrieval_keys)} examples from {retrieval_path}')

    def _get_ViT_mask(self, bbox, height, width):
        arr = np.zeros((self.patch_size, self.patch_size))
        x_min, y_min, x_max, y_max = bbox
        height_bins = np.linspace(0, height, self.patch_size)
        width_bins = np.linspace(0, width, self.patch_size)
        x_min, x_max = np.digitize(np.array([x_min, x_max]), width_bins)
        y_min, y_max = np.digitize(np.array([y_min, y_max]), height_bins)
        arr[y_min:y_max + 1, x_min:x_max] = 1

        return np.append(1, arr.flatten())
    
    def _load_dataset(self, dataset):
        
        retrieval_entries = []
        entries = []
        bad_pairs = 0
        i = 0
        for item in tqdm(dataset):
            
            i += 1
            if self.images_num is not None and i == self.images_num:
                break
            
            width = item['width']
            height = item['height']
            chunk = []
            retrieval_chunk = []
            visited_object_ids = set()
            for rel in item['relationships']:
                subject = rel['subject']
                object = rel['object']
                predicate = rel['predicate']
                s_bbox = (subject['x'], subject['y'], subject['x'] + subject['w'], subject['y'] + subject['h'])
                o_bbox = (object['x'], object['y'], object['x'] + object['w'], object['y'] + object['h'])

                seg_o = self._get_ViT_mask(o_bbox, height, width)
                seg_s = self._get_ViT_mask(s_bbox, height, width)
                if sum(seg_o[1:]) == 0 or sum(seg_s[1:]) == 0:
                    bad_pairs += 1
                    continue

                label = f"{subject['names'][0]}|||{predicate}|||{object['names'][0]}"
                chunk.append({"path_to_image": [item["image"], item["image"]], "question": "[obj] [obj] What is the relationship between these objects?", "vit_mask": [seg_o, seg_s], "answer": label})

                if object["object_id"] not in visited_object_ids:
                    visited_object_ids.add(object["object_id"])
                    retrieval_chunk.append({"path_to_image": item["image"], "vit_mask": seg_o, "answer": object['names'][0], "object_id": object["object_id"]})
                if subject["object_id"] not in visited_object_ids: 
                    retrieval_chunk.append({"path_to_image": item["image"], "vit_mask": seg_s, "answer": subject['names'][0], "object_id": subject["object_id"]})
                    visited_object_ids.add(subject["object_id"])
            
            entries.extend(chunk)
            retrieval_entries.extend(retrieval_chunk)

        self.retrieval_entries = retrieval_entries
        print(f"Skipped over {bad_pairs} bad pairs (no pixels)")
        return entries

    def collate_fn(self, batch):
        return {
            'path_to_image': [item["path_to_image"] for item in batch],
            'question': [item["question"] for item in batch],
            'vit_mask': [item["vit_mask"] for item in batch],
            "answer":  [item["answer"] for item in batch],
        }

    def eval_correctness(self, prediction, answer):
        correct = 1 if prediction == answer else 0
        score_dict = {}
        score_dict["score"] = correct
        return  score_dict

    def stats(self):
        return "Not Implemented"


    def find_topk_with_different_images(self, closest_indices, k):
    
        def select_different_images(closest_indices_row):
            selected_indices = []
            seen_images = set()

            for x in closest_indices_row.tolist():
                entry = self.retrieval_entries[self.retrieval_idx[x]]
                if entry["path_to_image"].filename not in seen_images:
                    selected_indices.append(x)
                    seen_images.add(entry["path_to_image"].filename)
                
                if len(selected_indices) >= k: 
                    break

            return selected_indices

        different_image_closest  = [select_different_images(closest_indices_row=closest_indices[i]) for i in range(len(closest_indices))]
        return torch.tensor( different_image_closest, device=closest_indices.device, dtype=torch.long)

    
    # Should take in argument k: how many closest objects to retrieve
    # and object features: the ViT features of query objects
    # Return: The k closest examples from self.entries according to cosine similarity
    # Note: features should be normalized so dot product == cosine similarity
    def retrieve_closest(self, object_features, k, train_phase = True, b_num=-1):
 
        dist_matrix = (object_features @ self.retrieval_keys.T)

        # If training, do not retrieve closest object to avoid label leakage
        if train_phase:
            closest_indices = torch.argsort(dist_matrix, axis = -1, descending=True)  # [:, :1+10*k]
            closest_indices = self.find_topk_with_different_images(closest_indices, k+1)[:, 1:k+1]
        else:
            closest_indices = torch.argsort(dist_matrix, axis = -1, descending=True) #[:, 0:k]
            closest_indices = self.find_topk_with_different_images(closest_indices, k)

        similarity_scores = [[round(dist_matrix[i, x].item(), 2) for x in closest_indices[i,:]] for i in range(len(closest_indices))]
        
        #  chunk.append({"id": item["id"], "path_to_image": item["image"], "question": random.choice(self.prompts), "vit_mask": seg, "answer": label, "original_segmentation": original_seg, 'bbox':bbox})

        def retrieve_info(x):
            entry = self.retrieval_entries[self.retrieval_idx[x]]
            label = self.retrieval_labels[x]
            path_to_image = entry["path_to_image"]
            vit_mask = entry["vit_mask"]
            object_id = entry["object_id"]
            return  { "path_to_image": path_to_image, "vit_mask": vit_mask, "answer": label, "object_id": object_id}

        retrieved_info = [[    retrieve_info(x) for x in closest_indices[i,:]    ] for i in range(len(closest_indices))]
        #retrieved_info = [[    self.entries[self.retrieval_idx[x]] for x in closest_indices[i,:]    ] for i in range(len(closest_indices))]
    
        return retrieved_info, similarity_scores

    def __str__(self):
        return f"Visual Genome dataset with {len(self.entries)} questions"


    def __len__(self):
        return len(self.entries)


    def __getitem__(self, index):
        entry = self.entries[index]

        return entry