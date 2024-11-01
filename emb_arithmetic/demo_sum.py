import yaml
from models.olive import OLIVE
from dataset.objectCOCO import COCOObjectDataset
import gradio as gr
import time
import numpy as np
import matplotlib.pyplot as plt
import skimage
import math
import torch
from PIL import Image

# from dataset.CXR8 import CXR8Dataset

with open("configs/config.yaml", 'r') as file:
    config = yaml.safe_load(file)

config['n_patches'] = 16
if "336" in config['vision_encoder']:
    config['n_patches'] = 24

if config["use_retrieval"]:
    # dataset = CXR8Dataset(config, split="train", n_patches=config["n_patches"])
    dataset = COCOObjectDataset(config, split="train", n_patches=config['n_patches'], max_examples_per_class=1000000)

model = None
old_config = None


def _get_ViT_mask(mask, height, width, output_height, output_width):
    pooled_mask = skimage.measure.block_reduce(mask, block_size=(
        math.floor(height / output_height), math.floor(width / output_width)), func=np.max)

    result_height, result_width = pooled_mask.shape
    # If the result is smaller than 16x16, pad it with zeros
    if result_height < output_height or result_width < output_width:
        pad_height = output_height - result_height
        pad_width = output_width - result_width
        pooled_mask = np.pad(pooled_mask, ((0, pad_height), (0, pad_width)), mode='constant')

    if result_height > output_height or result_width > output_width:
        pooled_mask = pooled_mask[:output_height, :output_width]

    assert pooled_mask.shape == (output_height, output_width)
    return torch.BoolTensor(np.append(1, pooled_mask.flatten()))


def sleep(im):
    time.sleep(2)
    ret = [im["background"]]
    for layer in im["layers"]:
        ret.append(layer)
    return ret


def generate_encoding(question, images, output_height, output_width):
    image = Image.open(images.root[0].image.path).convert('RGB')
    segmentations = [Image.open(x.image.path).convert('RGB') for x in images.root[1:]]

    seg_width, seg_height = image.size

    vit_masks = []

    cropped_images = []
    for segmentation in segmentations:
        seg = np.array(segmentation)
        if np.sum(seg, axis=None) == 0:
            continue
        else:

            mask = np.any(seg != [0, 0, 0], axis=-1)

            if config["crop_image"]:
                img = np.array(image)
                img[~mask] = np.array([255, 255, 255])

                # Find the indices of non-zero elements in the binary mask
                non_zero_indices = np.where(mask)

                # Get the minimum and maximum values along each axis
                min_x, min_y = np.min(non_zero_indices[1]), np.min(non_zero_indices[0])
                max_x, max_y = np.max(non_zero_indices[1]), np.max(non_zero_indices[0])

                img = img[min_y: max_y, min_x: max_x]

                cropped_image = Image.fromarray(np.uint8(img)).convert('RGB')
                cropped_images.append(cropped_image)

            vit_masks.append(_get_ViT_mask(mask, seg_height, seg_width, output_height, output_width))

    if len(vit_masks) > 0:
        vit_masks = torch.stack(vit_masks, axis=0)
    imgs = [image] * len(vit_masks) if len(vit_masks) > 0 else [image]

    if config['use_retrieval']:
        full_text_input, object_embeddings, labels, image_features, prompts, retrieved_masks, retrieved_images = (
            model.prepare_embeddings(vit_masks, imgs, [question], labels=None, return_retrieved_info=True,
                                     cropped_images=cropped_images))
        return full_text_input, object_embeddings, labels, image_features, prompts, retrieved_masks, retrieved_images
    else:
        full_text_input, object_embeddings, labels, image_features = model.prepare_embeddings(vit_masks, imgs,
                                                                                              [question], labels=None,
                                                                                              cropped_images=cropped_images)
        return full_text_input, object_embeddings, labels, image_features


def sum_encodings(images_1, images_2, backbone, use_retrieval, freeze_llm, subtract, chat_history):
    global model
    global old_config

    config['freeze_llm'] = freeze_llm
    config['llm_model'] = backbone
    config['task'] = "object_classification"
    config['use_retrieval'] = use_retrieval

    if "llama" or "gpt2" in backbone:
        if "336" in config["vision_encoder"]:
            output_width, output_height = 24, 24
        else:
            output_width, output_height = 16, 16

    elif "llava" in backbone:
        output_width, output_height = 24, 24

    else:
        output_width, output_height = 24, 24

    config['n_patches'] = output_width

    if old_config != config:
        if config['use_retrieval']:
            model = OLIVE(config, retrieval_fn=lambda x, y: dataset.retrieve_closest(x, config["retrieval_k"],
                                                                                     train_phase=False, b_num=y))
        else:
            model = OLIVE(config)
        model.load()
        model.eval()
        old_config = config.copy()

    question = "[obj] What is this?"

    if config['use_retrieval']:
        full_text_input_1, object_embeddings_1, labels_1, image_features_1, prompts_1, retrieved_masks_1, retrieved_images_1 = (
            generate_encoding(question, images_1, output_height, output_width))
        full_text_input_2, object_embeddings_2, labels_2, image_features_2, prompts_2, retrieved_masks_2, retrieved_images_2 = (
            generate_encoding(question, images_2, output_height, output_width))

        # Replace obj tokens with their object vector representation
        final_input_1, label_input_1, attention_mask_1 = model.embed_with_special_tokens(full_text_input_1,
                                                                                         object_embeddings_1,
                                                                                         labels=labels_1,
                                                                                         image_features=image_features_1)
        final_input_2, label_input_2, attention_mask_2 = model.embed_with_special_tokens(full_text_input_2,
                                                                                         object_embeddings_2,
                                                                                         labels=labels_2,
                                                                                         image_features=image_features_2)

        out_1 = model.llama_model.generate(inputs_embeds=final_input_1,
                                           attention_mask=attention_mask_1,
                                           max_new_tokens=100,
                                           top_p=0.0,
                                           top_k=1)
        out_2 = model.llama_model.generate(inputs_embeds=final_input_2,
                                           attention_mask=attention_mask_2,
                                           max_new_tokens=100,
                                           top_p=0.0,
                                           top_k=1)

        object_embeddings_sum = []
        for emb_1, emb_2 in zip(object_embeddings_1, object_embeddings_2):
            if subtract:
                emb_2 = torch.mul(emb_2, -1)
            object_embeddings_sum.append(torch.add(emb_1, emb_2))

        final_input_sum, label_input_sum, attention_mask_sum = model.embed_with_special_tokens(full_text_input_1,
                                                                                               object_embeddings_sum,
                                                                                               labels=labels_1,
                                                                                               image_features=image_features_1)

        out_sum = model.llama_model.generate(inputs_embeds=final_input_sum,
                                             attention_mask=attention_mask_sum,
                                             max_new_tokens=100,
                                             top_p=0.0,
                                             top_k=1)

        decoded_output_1 = model.tokenizer.batch_decode(out_1, skip_special_tokens=True)
        decoded_output_2 = model.tokenizer.batch_decode(out_2, skip_special_tokens=True)
        decoded_output_sum = model.tokenizer.batch_decode(out_sum, skip_special_tokens=True)

        if len(decoded_output_1) == 1:
            decoded_output_1 = decoded_output_1[0]
        if len(decoded_output_2) == 1:
            decoded_output_2 = decoded_output_2[0]
        if len(decoded_output_sum) == 1:
            decoded_output_sum = decoded_output_sum[0]

        operand = " + "
        if subtract:
            operand = " - "

        chat_history.append((question, decoded_output_1 + operand + decoded_output_2 + " = " + decoded_output_sum))
        retrieval_images = [Image.open(images[0][x]) for x in range(len(retrieved_images_1[0]))]
        retrieval_images.extend([Image.open(images[0][x]) for x in range(len(retrieved_images_2[0]))])
        return chat_history, retrieval_images

    else:
        full_text_input_1, object_embeddings_1, labels_1, image_features_1 = generate_encoding(question, images_1,
                                                                                               output_height,
                                                                                               output_width)
        full_text_input_2, object_embeddings_2, labels_2, image_features_2 = generate_encoding(question, images_2,
                                                                                               output_height,
                                                                                               output_width)

        # Replace obj tokens with their object vector representation
        final_input_1, label_input_1, attention_mask_1 = model.embed_with_special_tokens(full_text_input_1,
                                                                                         object_embeddings_1,
                                                                                         labels=labels_1,
                                                                                         image_features=image_features_1)
        final_input_2, label_input_2, attention_mask_2 = model.embed_with_special_tokens(full_text_input_2,
                                                                                         object_embeddings_2,
                                                                                         labels=labels_2,
                                                                                         image_features=image_features_2)

        out_1 = model.llama_model.generate(inputs_embeds=final_input_1,
                                           attention_mask=attention_mask_1,
                                           max_new_tokens=100,
                                           top_p=0.0,
                                           top_k=1)
        out_2 = model.llama_model.generate(inputs_embeds=final_input_2,
                                           attention_mask=attention_mask_2,
                                           max_new_tokens=100,
                                           top_p=0.0,
                                           top_k=1)

        object_embeddings_sum = []
        for emb_1, emb_2 in zip(object_embeddings_1, object_embeddings_2):
            if subtract:
                emb_2 = torch.mul(emb_2, -1)
            object_embeddings_sum.append(torch.add(emb_1, emb_2))

        final_input_sum, label_input_sum, attention_mask_sum = model.embed_with_special_tokens(full_text_input_1,
                                                                                               object_embeddings_sum,
                                                                                               labels=labels_1,
                                                                                               image_features=image_features_1)

        out_sum = model.llama_model.generate(inputs_embeds=final_input_sum,
                                             attention_mask=attention_mask_sum,
                                             max_new_tokens=100,
                                             top_p=0.0,
                                             top_k=1)

        decoded_output_1 = model.tokenizer.batch_decode(out_1, skip_special_tokens=True)
        decoded_output_2 = model.tokenizer.batch_decode(out_2, skip_special_tokens=True)
        decoded_output_sum = model.tokenizer.batch_decode(out_sum, skip_special_tokens=True)

        if len(decoded_output_1) == 1:
            decoded_output_1 = decoded_output_1[0]
        if len(decoded_output_2) == 1:
            decoded_output_2 = decoded_output_2[0]
        if len(decoded_output_sum) == 1:
            decoded_output_sum = decoded_output_sum[0]

        operand = " + "
        if subtract:
            operand = " - "

        chat_history.append((question, decoded_output_1 + operand + decoded_output_2 + " = " + decoded_output_sum))
        return chat_history, None


with gr.Blocks(title="Olive", theme=gr.themes.Base()).queue() as demo:
    with gr.Row():
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    im_1 = gr.ImageEditor(
                        type="pil"
                    )
                    with gr.Row():
                        gallery_1 = gr.Gallery(
                            label="Segmentations", show_label=False, elem_id="gallery_1"
                            , columns=[3], rows=[1], object_fit="contain", height=200)

                    im_2 = gr.ImageEditor(
                        type="pil"
                    )
                    with gr.Row():
                        gallery_2 = gr.Gallery(
                            label="Segmentations", show_label=False, elem_id="gallery_2"
                            , columns=[3], rows=[1], object_fit="contain", height=200)

                with gr.Column():
                    chatbot = gr.Chatbot(elem_id="chatbot", label="OLIVE Chatbot", height=300)
                    with gr.Row():
                        with gr.Column(scale=1, min_width=50):
                            submit_btn = gr.Button(value="Sum", variant="primary")
                    retrieval_gallery = gr.Gallery(
                        label="Retrieved Images", show_label=True, elem_id="gallery3"
                        , columns=[5], rows=[1], object_fit="contain", height=100)

                    backbone = gr.Dropdown(["llava-hf/llava-1.5-7b-hf", "meta-llama/Llama-2-7b-chat-hf", "gpt2"],
                                           label="Decoder Backbone", info="Backbone Frozen LLM/VLM",
                                           value="meta-llama/Llama-2-7b-chat-hf")

                    freeze_llm = gr.Checkbox(label="freeze llm", info="Freeze llm weights", value=False)
                    use_retrieval = gr.Checkbox(label="use retrieval", info="Use retrieval to understand prediction",
                                                value=False)
                    subtract = gr.Checkbox(label="subtract", info="Subtract object encodings", value=False)

        im_1.change(sleep, outputs=[gallery_1], inputs=im_1)
        im_2.change(sleep, outputs=[gallery_2], inputs=im_2)

    submit_btn.click(fn=sum_encodings,
                     inputs=[gallery_1, gallery_2, backbone, use_retrieval, freeze_llm, subtract, chatbot],
                     outputs=[chatbot, retrieval_gallery],
                     show_progress=True, queue=True)

demo.launch(share=True, inbrowser=True)
