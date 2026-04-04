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

img_path = os.path.join('../flickr30k-images', os.listdir('../flickr30k-images')[0])
img = Image.open(img_path).convert('RGB')
image_tensor = process_images([img], image_processor, model.config).to(device, dtype=torch.float16)

prompt = DEFAULT_IMAGE_TOKEN + '\nWhat is in this image?'
conv = conv_templates['llava_v0'].copy()
conv.append_message(conv.roles[0], prompt)
conv.append_message(conv.roles[1], None)
input_ids = tokenizer_image_token(
    conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
).unsqueeze(0).to(device)

with torch.inference_mode():
    out = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[img.size],
        max_new_tokens=100,
        do_sample=False,
    )

print(f'Input shape:  {input_ids.shape}')
print(f'Output shape: {out.shape}')
print(f'\n--- Sliced (old way) ---')
print(repr(tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)))
print(f'\n--- Full output (new way) ---')
print(repr(tokenizer.decode(out[0], skip_special_tokens=True)))
