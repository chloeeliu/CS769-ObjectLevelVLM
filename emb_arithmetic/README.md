# Embedding Arithmetic

## Overview
This repository contains the code for performing object embedding arithmetic experiments.

## Notable changes
Two new demos are created to allow uses to test embedding arithmetic qualitatively. 
One demo script allows users to upload two images, draw masks for each, and finally decode the sum of the two object embeddings.

The second script similarly allows uploading three images and masks, before trying to construct an analogy with the word embeddings.

## Setup
Setup is identical to the original OLIVE repository, which recommends setting up an anaconda environment:
```
conda env create -f environment.yml
conda activate olive
```

The necessary datasets can then be downloaded and preprocessed using the same OLIVE setup script:
```
python setup/setup.py
```

If you would like to use OLIVE with retrieval, then prepare the retrieval set:
```
python retrieve.py --train --config <path_to_config_file>
```

### Model checkpoints

The pretrained OLIVE model can be downloaded from HuggingFace using `git lfs` (make sure to update the config file with the model path).

For the reported experiments, [OLIVE-G-Classification](https://huggingface.co/tossowski/OLIVE-G-Classification) was used for all settings.

### Config files
An example config used for the experiments detailed in the report can be found in `configs/config.yaml`. Ensure that file paths are correct for the model and retrieval set you would like to use.

## Sum demo
```
python demo_sum.py
```
This will open the gradio interface to allow uploading images and drawing masks. You can select to use retrieval if an image retrieval set has been created.

After uploading your images and clicking the `Sum` button, the OLIVE model will generate labels for each specified object and the sum of their object embeddings.

Finally, the `Subtract` check box allows the user to compute the difference `(object_1 - object_2)` instead of sum.

## Analogy demo
```
python demo_analogy.py
```
Similarly, this will open the gradio interface to upload three images and draw the corresponding masks.

Clicking the `Analogy` button will use the OLIVE model to generate labels for each object and the arithmetic analogy of their embeddings: `(object_1 -> object_2, as object_3 -> ____)`.