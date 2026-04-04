# -*- coding: utf-8 -*-
"""
Generate tags AND captions for all images using LLaVA 1.5.
Supports sharded parallel execution — run one process per GPU for 4x speedup.

Single GPU usage:
    CUDA_VISIBLE_DEVICES=0 python tag.py --shard 0 --num_shards 4 ...

After all 4 shards finish, merge:
    python tag.py --merge_only
"""
import argparse
import json
import os
import requests

import torch
from PIL import Image
from io import BytesIO
from tqdm import tqdm

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import (IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
                              DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN)
from llava.conversation import conv_templates, SeparatorStyle

TAG_PROMPT = (
    'List 5 short descriptive tags for this image. '
    'Format your response exactly like this example: '
    '#outdoor scene #people walking #city street #casual clothing #daytime. '
    'Only output the tags, nothing else.'
)
CAPTION_PROMPT = (
    'Describe what is happening in this image in one complete sentence.'
)


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        return Image.open(BytesIO(response.content)).convert('RGB')
    return Image.open(image_file).convert('RGB')


def build_prompt(text_prompt, model_config):
    mm_use_im_start_end = getattr(model_config, 'mm_use_im_start_end', False)
    if mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + text_prompt
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + text_prompt
    conv = conv_templates['llava_v0'].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt(), conv


def run_inference(model, tokenizer, image_tensor, image_size, prompt, conv, device):
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=False,
            max_new_tokens=128,
            use_cache=True,
        )
    # LLaVA 1.5 with image_sizes returns only the newly generated tokens,
    # NOT input+new. Decode output_ids directly (not sliced).
    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if output.endswith(stop_str):
        output = output[:-len(stop_str)].strip()
    return output


def shard_paths(tag_path, caption_path, shard):
    base_tag = tag_path.replace('.json', f'_shard{shard}.json')
    base_cap = caption_path.replace('.json', f'_shard{shard}.json')
    return base_tag, base_cap


def merge(tag_path, caption_path, num_shards):
    merged_tags, merged_caps = {}, {}
    for s in range(num_shards):
        st, sc = shard_paths(tag_path, caption_path, s)
        if os.path.exists(st):
            merged_tags.update(json.load(open(st, 'r', encoding='utf-8')))
        if os.path.exists(sc):
            merged_caps.update(json.load(open(sc, 'r', encoding='utf-8')))
    json.dump(merged_tags, open(tag_path,     'w', encoding='utf-8'), ensure_ascii=False)
    json.dump(merged_caps, open(caption_path, 'w', encoding='utf-8'), ensure_ascii=False)
    print(f'Merged {len(merged_tags)} tags and {len(merged_caps)} captions.')
    print(f'Saved: {tag_path}')
    print(f'Saved: {caption_path}')


def eval_model(img_dir_path, tag_path, caption_path, model_path,
               shard=0, num_shards=1):
    # Each shard writes to its own file
    shard_tag_path, shard_cap_path = shard_paths(tag_path, caption_path, shard)

    # Load model on the single visible GPU
    model_name = get_model_name_from_path(os.path.expanduser(model_path))
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=os.path.expanduser(model_path),
        model_base=None,
        model_name=model_name,
        device_map='auto',
    )
    device = next(model.parameters()).device
    print(f'[Shard {shard}] Model loaded on {device}')

    # Resume support
    img_to_tags     = json.load(open(shard_tag_path, 'r', encoding='utf-8')) if os.path.exists(shard_tag_path) else {}
    img_to_captions = json.load(open(shard_cap_path, 'r', encoding='utf-8')) if os.path.exists(shard_cap_path) else {}
    already_done = set(img_to_tags.keys()) & set(img_to_captions.keys())

    all_imgs = sorted([f for f in os.listdir(img_dir_path) if f.lower().endswith('.jpg')])
    # Assign this shard's slice
    shard_imgs = all_imgs[shard::num_shards]
    pending    = [f for f in shard_imgs if f[:-4] not in already_done]
    print(f'[Shard {shard}] {len(already_done)} done, {len(pending)} remaining of {len(shard_imgs)}')

    tag_prompt_str,     tag_conv     = build_prompt(TAG_PROMPT,     model.config)
    caption_prompt_str, caption_conv = build_prompt(CAPTION_PROMPT, model.config)

    for i, img_name in enumerate(tqdm(pending, desc=f'shard-{shard}')):
        img_id   = img_name[:-4]
        img_file = os.path.join(img_dir_path, img_name)
        try:
            image        = load_image(img_file)
            image_size   = image.size
            image_tensor = process_images([image], image_processor, model.config)
            if isinstance(image_tensor, list):
                image_tensor = [t.to(device, dtype=torch.float16) for t in image_tensor]
            else:
                image_tensor = image_tensor.to(device, dtype=torch.float16)

            img_to_tags[img_id]     = run_inference(model, tokenizer, image_tensor, image_size,
                                                    tag_prompt_str,     tag_conv,     device)
            img_to_captions[img_id] = run_inference(model, tokenizer, image_tensor, image_size,
                                                    caption_prompt_str, caption_conv, device)
            print(f'[{img_id}] TAGS: {img_to_tags[img_id]}')
            print(f'[{img_id}] CAPTION: {img_to_captions[img_id]}')
        except Exception as e:
            print(f'[WARN] Skipping {img_name}: {e}')
            continue

        if (i + 1) % 100 == 0:
            json.dump(img_to_tags,     open(shard_tag_path, 'w', encoding='utf-8'), ensure_ascii=False)
            json.dump(img_to_captions, open(shard_cap_path, 'w', encoding='utf-8'), ensure_ascii=False)
            print(f'  [shard {shard} checkpoint] {i+1} images saved')

    json.dump(img_to_tags,     open(shard_tag_path, 'w', encoding='utf-8'), ensure_ascii=False)
    json.dump(img_to_captions, open(shard_cap_path, 'w', encoding='utf-8'), ensure_ascii=False)
    print(f'[Shard {shard}] Done.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir',      default='../flickr30k-images')
    parser.add_argument('--tag_path',     default='data/tag_responses_full.json')
    parser.add_argument('--caption_path', default='data/caption_responses_full.json')
    parser.add_argument('--model_path',   default='output/LLaVA-13B-v1.1')
    parser.add_argument('--shard',        type=int, default=0,
                        help='Which shard to process (0-indexed)')
    parser.add_argument('--num_shards',   type=int, default=1,
                        help='Total number of shards (= number of GPUs)')
    parser.add_argument('--merge_only',   action='store_true',
                        help='Just merge existing shard files, no inference')
    args = parser.parse_args()

    if args.merge_only:
        merge(args.tag_path, args.caption_path, args.num_shards)
    else:
        eval_model(args.img_dir, args.tag_path, args.caption_path,
                   args.model_path, args.shard, args.num_shards)
