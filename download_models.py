import os
from huggingface_hub import snapshot_download

models = [
    'openai/clip-vit-base-patch32',
    'nvidia/groupvit-gcc-yfcc',
    'kakaobrain/align-base',
    'CIDAS/clipseg-rd64-refined',
]

for m in models:
    name = m.split('/')[1]
    print(f'Downloading {name}...')
    snapshot_download(repo_id=m, local_dir=f'{os.path.expanduser("~/models")}/{name}')
    print(f'Done: {name}')
