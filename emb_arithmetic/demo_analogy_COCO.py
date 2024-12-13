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
    train_dataset = COCOObjectDataset(config, split="train", n_patches=config['n_patches'], max_examples_per_class=1000000)

dataset = COCOObjectDataset(config, split="val", n_patches=config['n_patches'],
                            max_examples_per_class=config["examples_per_class"])

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


def generate_encoding(question, image, vit_masks, output_height, output_width):
    if config['use_retrieval']:
        full_text_input, object_embeddings, labels, image_features, prompts, retrieved_masks, retrieved_images = (
            model.prepare_embeddings(vit_masks, image, [question], labels=None, return_retrieved_info=True))
        return full_text_input, object_embeddings, labels, image_features, prompts, retrieved_masks, retrieved_images
    else:
        full_text_input, object_embeddings, labels, image_features = model.prepare_embeddings(vit_masks, image,
                                                                                              [question], labels=None)
        return full_text_input, object_embeddings, labels, image_features


def compute_analogy(backbone, use_retrieval, freeze_llm, chat_history):
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
            model = OLIVE(config, retrieval_fn=lambda x, y: train_dataset.retrieve_closest(x, config["retrieval_k"],
                                                                                     train_phase=False, b_num=y))
        else:
            model = OLIVE(config)
        model.load()
        model.eval()
        old_config = config.copy()

    question = "[obj] What is this?"

    sample_idx = torch.randint(len(dataset), size=(3,))
    sample_1 = dataset[sample_idx[0]]
    sample_2 = dataset[sample_idx[1]]
    sample_3 = dataset[sample_idx[2]]

    images_1 = [sample_1['path_to_image']]
    images_2 = [sample_2['path_to_image']]
    images_3 = [sample_3['path_to_image']]

    images_opened_1 = [Image.open(images_1[0])]
    images_opened_2 = [Image.open(images_2[0])]
    images_opened_3 = [Image.open(images_3[0])]

    masks_1 = [torch.BoolTensor(sample_1["vit_mask"]).to(config["device"])]
    masks_2 = [torch.BoolTensor(sample_2["vit_mask"]).to(config["device"])]
    masks_3 = [torch.BoolTensor(sample_3["vit_mask"]).to(config["device"])]

    if config['use_retrieval']:
        full_text_input_1, object_embeddings_1, labels_1, image_features_1, prompts_1, retrieved_masks_1, retrieved_images_1 = (
            generate_encoding(question, images_1, masks_1, output_height, output_width))
        full_text_input_2, object_embeddings_2, labels_2, image_features_2, prompts_2, retrieved_masks_2, retrieved_images_2 = (
            generate_encoding(question, images_2, masks_2, output_height, output_width))
        full_text_input_3, object_embeddings_3, labels_3, image_features_3, prompts_3, retrieved_masks_3, retrieved_images_3 = (
            generate_encoding(question, images_3, masks_3, output_height, output_width))

        # Replace obj tokens with their object vector representation
        final_input_1, label_input_1, attention_mask_1 = model.embed_with_special_tokens(full_text_input_1,
                                                                                         object_embeddings_1,
                                                                                         labels=labels_1,
                                                                                         image_features=image_features_1)
        final_input_2, label_input_2, attention_mask_2 = model.embed_with_special_tokens(full_text_input_2,
                                                                                         object_embeddings_2,
                                                                                         labels=labels_2,
                                                                                         image_features=image_features_2)
        final_input_3, label_input_3, attention_mask_3 = model.embed_with_special_tokens(full_text_input_3,
                                                                                         object_embeddings_3,
                                                                                         labels=labels_3,
                                                                                         image_features=image_features_3)

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
        out_3 = model.llama_model.generate(inputs_embeds=final_input_3,
                                           attention_mask=attention_mask_3,
                                           max_new_tokens=100,
                                           top_p=0.0,
                                           top_k=1)

        object_embeddings_analogy = []
        for emb_1, emb_2, emb_3 in zip(object_embeddings_1, object_embeddings_2, object_embeddings_3):
            object_embeddings_analogy.append(torch.add(torch.sub(emb_2, emb_1), emb_3))

        final_input_analogy, label_input_analogy, attention_mask_analogy = model.embed_with_special_tokens(
            full_text_input_1,
            object_embeddings_analogy,
            labels=labels_1,
            image_features=image_features_1)

        out_analogy = model.llama_model.generate(inputs_embeds=final_input_analogy,
                                                 attention_mask=attention_mask_analogy,
                                                 max_new_tokens=100,
                                                 top_p=0.0,
                                                 top_k=1)

        decoded_output_1 = model.tokenizer.batch_decode(out_1, skip_special_tokens=True)
        decoded_output_2 = model.tokenizer.batch_decode(out_2, skip_special_tokens=True)
        decoded_output_3 = model.tokenizer.batch_decode(out_3, skip_special_tokens=True)
        decoded_output_analogy = model.tokenizer.batch_decode(out_analogy, skip_special_tokens=True)

        if len(decoded_output_1) == 1:
            decoded_output_1 = decoded_output_1[0]
        if len(decoded_output_2) == 1:
            decoded_output_2 = decoded_output_2[0]
        if len(decoded_output_3) == 1:
            decoded_output_3 = decoded_output_3[0]
        if len(decoded_output_analogy) == 1:
            decoded_output_analogy = decoded_output_analogy[0]

        chat_history.append(
            (question, f'{decoded_output_1} -> {decoded_output_2} as  {decoded_output_3} -> {decoded_output_analogy}'))
        retrieval_images = [Image.open(images[0][x]) for x in range(len(retrieved_images_1[0]))]
        retrieval_images.extend([Image.open(images[0][x]) for x in range(len(retrieved_images_2[0]))])
        retrieval_images.extend([Image.open(images[0][x]) for x in range(len(retrieved_images_3[0]))])
        return images_opened_1, images_opened_2, images_opened_3, chat_history, retrieval_images

    else:
        full_text_input_1, object_embeddings_1, labels_1, image_features_1 = generate_encoding(question,
                                                                                               images_1,
                                                                                               masks_1,
                                                                                               output_height,
                                                                                               output_width)
        full_text_input_2, object_embeddings_2, labels_2, image_features_2 = generate_encoding(question,
                                                                                               images_2,
                                                                                               masks_2,
                                                                                               output_height,
                                                                                               output_width)
        full_text_input_3, object_embeddings_3, labels_3, image_features_3 = generate_encoding(question,
                                                                                               images_3,
                                                                                               masks_3,
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
        final_input_3, label_input_3, attention_mask_3 = model.embed_with_special_tokens(full_text_input_3,
                                                                                         object_embeddings_3,
                                                                                         labels=labels_3,
                                                                                         image_features=image_features_3)

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
        out_3 = model.llama_model.generate(inputs_embeds=final_input_3,
                                           attention_mask=attention_mask_3,
                                           max_new_tokens=100,
                                           top_p=0.0,
                                           top_k=1)

        object_embeddings_analogy = []
        for emb_1, emb_2, emb_3 in zip(object_embeddings_1, object_embeddings_2, object_embeddings_3):
            object_embeddings_analogy.append(torch.add(torch.sub(emb_2, emb_1), emb_3))

        final_input_analogy, label_input_analogy, attention_mask_analogy = model.embed_with_special_tokens(
            full_text_input_1,
            object_embeddings_analogy,
            labels=labels_1,
            image_features=image_features_1)

        out_analogy = model.llama_model.generate(inputs_embeds=final_input_analogy,
                                                 attention_mask=attention_mask_analogy,
                                                 max_new_tokens=100,
                                                 top_p=0.0,
                                                 top_k=1)

        decoded_output_1 = model.tokenizer.batch_decode(out_1, skip_special_tokens=True)
        decoded_output_2 = model.tokenizer.batch_decode(out_2, skip_special_tokens=True)
        decoded_output_3 = model.tokenizer.batch_decode(out_3, skip_special_tokens=True)
        decoded_output_analogy = model.tokenizer.batch_decode(out_analogy, skip_special_tokens=True)

        if len(decoded_output_1) == 1:
            decoded_output_1 = decoded_output_1[0]
        if len(decoded_output_2) == 1:
            decoded_output_2 = decoded_output_2[0]
        if len(decoded_output_3) == 1:
            decoded_output_3 = decoded_output_3[0]
        if len(decoded_output_analogy) == 1:
            decoded_output_analogy = decoded_output_analogy[0]

        chat_history.append(
            (question, f'{decoded_output_1} -> {decoded_output_2} as  {decoded_output_3} -> {decoded_output_analogy}'))
        return images_opened_1, images_opened_2, images_opened_3, chat_history, None


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

                    im_3 = gr.ImageEditor(
                        type="pil"
                    )
                    with gr.Row():
                        gallery_3 = gr.Gallery(
                            label="Segmentations", show_label=False, elem_id="gallery_3"
                            , columns=[3], rows=[1], object_fit="contain", height=200)

                with gr.Column():
                    chatbot = gr.Chatbot(elem_id="chatbot", label="OLIVE Chatbot", height=300)
                    with gr.Row():
                        with gr.Column(scale=1, min_width=50):
                            submit_btn = gr.Button(value="Analogy", variant="primary")
                    retrieval_gallery = gr.Gallery(
                        label="Retrieved Images", show_label=True, elem_id="gallery4"
                        , columns=[5], rows=[1], object_fit="contain", height=100)

                    backbone = gr.Dropdown(["llava-hf/llava-1.5-7b-hf", "meta-llama/Llama-2-7b-chat-hf", "gpt2"],
                                           label="Decoder Backbone", info="Backbone Frozen LLM/VLM",
                                           value="meta-llama/Llama-2-7b-chat-hf")

                    freeze_llm = gr.Checkbox(label="freeze llm", info="Freeze llm weights", value=False)
                    use_retrieval = gr.Checkbox(label="use retrieval", info="Use retrieval to understand prediction",
                                                value=False)

        im_1.change(sleep, outputs=[gallery_1], inputs=im_1)
        im_2.change(sleep, outputs=[gallery_2], inputs=im_2)
        im_3.change(sleep, outputs=[gallery_3], inputs=im_3)

    submit_btn.click(fn=compute_analogy,
                     inputs=[backbone, use_retrieval, freeze_llm, chatbot],
                     outputs=[gallery_1, gallery_2, gallery_3, chatbot, retrieval_gallery],
                     show_progress=True, queue=True)

demo.launch(share=True, inbrowser=True)
