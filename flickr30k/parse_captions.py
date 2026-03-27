"""
Converts the Flickr30K HTML captions file to the tab-separated format
expected by combine.py:  image_filename#index\tcaption
"""

import re

input_file = 'data/captions.html'   # path to the downloaded HTML file
output_file = 'data/30k_captions.txt'

with open(input_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Strip HTML tags
text = re.sub(r'<[^>]+>', '', content)

lines = [l.strip() for l in text.splitlines() if l.strip()]

with open(output_file, 'w', encoding='utf-8') as out:
    img = None
    idx = 0
    for line in lines:
        if line.endswith('.jpg'):
            img = line
            idx = 0
        elif img:
            out.write(f'{img}#{idx}\t{line}\n')
            idx += 1
