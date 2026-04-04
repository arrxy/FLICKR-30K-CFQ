import os
import torch
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

model_path = 'output/llava-v1.5-13b'
model_name = get_model_name_from_path(model_path)
print(f'Model name detected: {model_name}')

tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, model_name)
device = next(model.parameters()).device
print(f'Device: {device}')

img_dir = '../flickr30k-images'
img_file = os.path.join(img_dir, os.listdir(img_dir)[0])
print(f'Image: {img_file}')
img = Image.open(img_file).convert('RGB')

image_tensor = process_images([img], image_processor, model.config)
image_tensor = image_tensor.to(device, dtype=torch.float16)

# Try both conv templates
for conv_mode in ['llava_v1', 'vicuna_v1', 'llava_v0']:
    try:
        prompt = DEFAULT_IMAGE_TOKEN + '\nWhat is in this image?'
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(device)

        with torch.inference_mode():
            out = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[img.size],   # required for LLaVA 1.5
                max_new_tokens=100,
                do_sample=False,
            )

        print(f'\n[{conv_mode}] input shape: {input_ids.shape}, output shape: {out.shape}')
        print(f'Full raw output ids: {out[0].tolist()[-20:]}')   # last 20 tokens
        print(f'New tokens: {out[0, input_ids.shape[1]:].tolist()}')
        print(f'Decoded: {tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)}')
    except Exception as e:
        print(f'[{conv_mode}] Error: {e}')
