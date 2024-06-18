# Text-to-Image-by-merging-Dreambooth-and-LoRa 

This project aims to generate high-quality images of huskies by combining the Dreambooth and LoRa techniques. By fine-tuning with a dataset of husky photos, we achieve detailed and realistic images. All my results are based on fine-tuning [https://huggingface.co/runwayml/stable-diffusion-v1-5](https://huggingface.co/docs/diffusers/en/using-diffusers/sdxl) model.

## Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Results](#results)
- [References](#contributing)
  
## Introduction
This repository contains the implementation of a text-to-image model that leverages Dreambooth and LoRa for fine-tuning with specific images. The focus is on generating husky images with improved quality by using detailed prompts.

## Prerequisites
- Python 3.8+
- PyTorch 1.9+
- CUDA 10.2 or higher for GPU support
- Git

## Usage
### Preparing the Dataset
Collect and organize your husky photos into a dataset directory. Ensure that your images are high quality and well-labeled.

### Fine-tuning with Dreambooth and LoRa
Run the training script with the following command:

!accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path="your_pretrained_model" \
  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
  --dataset_name="husky" \
  --output_dir="your_output_dir" \
  --caption_column="prompt"\
  --mixed_precision="fp16" \
  --instance_prompt="a photo of husky" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=3 \
  --gradient_checkpointing \
  --learning_rate=1e-4 \
  --snr_gamma=5.0 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --max_train_steps=300 \
  --checkpointing_steps=717 \
  --seed="0"

## Results
We display the results using a range of training samples and images from different huskys. Here is the AI generated Husky Images. In text-to-image, the prompt is very priority. The better you are at prompting, the better the AI will generate husky images. 

![download](https://github.com/Ye-Bhone-Lin/Text-to-Image-by-merging-Dreambooth-and-LoRa/assets/106800189/9dad6cad-f76b-46e8-9956-ec18a77ef380) 
Prompt: A husky in the basket

![download (2)](https://github.com/Ye-Bhone-Lin/Text-to-Image-by-merging-Dreambooth-and-LoRa/assets/106800189/301c37ee-62e8-48a4-88bd-782880c4022a)
Prompt: the husky in captain America outfit, 16k

![download (3)](https://github.com/Ye-Bhone-Lin/Text-to-Image-by-merging-Dreambooth-and-LoRa/assets/106800189/68b9be6a-2176-4b4c-8ed4-f218cb3475fc)
Prompt: (ISOLATED ON A PRISTINE WHITE BACKGROUND: 1.5), T-SHIRT DESIGN, VECTOR ART, CONTOUR, Sad husky with a white background.

### References:

Parashar, S. (2023, November 18). Dreambooth: Fine-tuning text-to-image diffusion models for subject-driven generation. Medium. https://medium.com/@sankalpparasharblog/dreambooth-fine-tuning-text-to-image-diffusion-models-for-subject-driven-generation-8f6664d57aaf

Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., & Aberman, K. (2022). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation. arXiv. https://arxiv.org/abs/2208.12242

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv. https://arxiv.org/abs/2106.09685




