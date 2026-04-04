import os
import torch
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

model_path = 'output/llava-v1.5-13b'
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, model_name)
device = next(model.parameters()).device
print(f'Model loaded on {device}')

img_dir = '../flickr30k-images'
img_file = os.path.join(img_dir, os.listdir(img_dir)[0])
print(f'Testing on: {img_file}')
img = Image.open(img_file).convert('RGB')
image_size = img.size
image_tensor = process_images([img], image_processor, model.config).to(device, dtype=torch.float16)

for prompt_text in [
    'Describe this image in one sentence.',
    'What is happening in this image?',
    'List 5 keywords describing this image, each starting with #',
]:
    prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt_text
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
            image_sizes=[image_size],
            max_new_tokens=200,
            do_sample=False,
        )
    result = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
    print(f'\nPROMPT: {prompt_text}')
    print(f'OUTPUT: {result}')
