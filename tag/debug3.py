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
print(f'Model dtype: {next(model.parameters()).dtype}')
print(f'Model class: {model.__class__.__name__}')

# Test 1: text only (no image)
print('\n--- Test 1: Text only (no image) ---')
text_ids = tokenizer('The sky is', return_tensors='pt').input_ids.to(device)
with torch.inference_mode():
    out = model.generate(text_ids, max_new_tokens=20, do_sample=False)
print(f'Input shape: {text_ids.shape}, Output shape: {out.shape}')
print(f'Decoded: {tokenizer.decode(out[0, text_ids.shape[1]:], skip_special_tokens=True)}')

# Test 2: with image using llava_v0
print('\n--- Test 2: With image (llava_v0) ---')
img_path = os.path.join('../flickr30k-images', os.listdir('../flickr30k-images')[0])
img = Image.open(img_path).convert('RGB')
image_tensor = process_images([img], image_processor, model.config).to(device, dtype=torch.float16)

prompt = DEFAULT_IMAGE_TOKEN + '\nWhat is in this image?'
conv = conv_templates['llava_v0'].copy()
conv.append_message(conv.roles[0], prompt)
conv.append_message(conv.roles[1], None)
full_prompt = conv.get_prompt()
print(f'Prompt: {repr(full_prompt)}')

input_ids = tokenizer_image_token(
    full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
).unsqueeze(0).to(device)
print(f'Input ids: {input_ids[0].tolist()}')  # print all token ids

with torch.inference_mode():
    out = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[img.size],
        max_new_tokens=50,
        do_sample=False,
    )
print(f'Input shape: {input_ids.shape}, Output shape: {out.shape}')
print(f'Output ids (last 20): {out[0][-20:].tolist()}')
print(f'Decoded: {tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)}')

# Test 3: check logits for first token
print('\n--- Test 3: First token logits ---')
with torch.inference_mode():
    logits = model(input_ids, images=image_tensor).logits
print(f'Logits shape: {logits.shape}')
top5 = torch.topk(logits[0, -1], 5)
print('Top 5 next tokens:')
for score, idx in zip(top5.values, top5.indices):
    print(f'  {repr(tokenizer.decode([idx.item()]))}: {score.item():.3f}')
