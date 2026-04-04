import os
import torch
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

model_path = 'output/llava-v1.5-13b'
tokenizer, model, image_processor, _ = load_pretrained_model(
    model_path, None, get_model_name_from_path(model_path))
device = next(model.parameters()).device
print(f'Device: {device}')

img_path = os.path.join('../flickr30k-images', os.listdir('../flickr30k-images')[0])
img = Image.open(img_path).convert('RGB')
print(f'Image size: {img.size}')

image_tensor = process_images([img], image_processor, model.config)
print(f'image_tensor type: {type(image_tensor)}')
if hasattr(image_tensor, 'shape'):
    print(f'image_tensor shape: {image_tensor.shape}')
image_tensor = image_tensor.to(device, dtype=torch.float16)

prompt = DEFAULT_IMAGE_TOKEN + '\nWhat is in this image?'
conv = conv_templates['llava_v0'].copy()
conv.append_message(conv.roles[0], prompt)
conv.append_message(conv.roles[1], None)
input_ids = tokenizer_image_token(
    conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
).unsqueeze(0).to(device)
print(f'input_ids shape: {input_ids.shape}')

# Without image_sizes
print('\n--- Without image_sizes ---')
with torch.inference_mode():
    out = model.generate(input_ids, images=image_tensor, max_new_tokens=50, do_sample=False)
print(f'output shape: {out.shape}')
print(f'new tokens: {out[0, input_ids.shape[1]:].tolist()}')
print(f'decoded: {tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)}')

# With image_sizes
print('\n--- With image_sizes ---')
with torch.inference_mode():
    out2 = model.generate(input_ids, images=image_tensor, image_sizes=[img.size],
                          max_new_tokens=50, do_sample=False)
print(f'output shape: {out2.shape}')
print(f'new tokens: {out2[0, input_ids.shape[1]:].tolist()}')
print(f'decoded: {tokenizer.decode(out2[0, input_ids.shape[1]:], skip_special_tokens=True)}')
