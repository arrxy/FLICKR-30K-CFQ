import json, os

total = 0
for s in range(4):
    tf = f'/home/cc/Flickr30K-CFQ/tag/data/tag_responses_full_shard{s}.json'
    n = len(json.load(open(tf))) if os.path.exists(tf) else 0
    print(f'Shard {s}: {n} / 7946')
    total += n
print(f'Total: {total} / 31783 ({100*total//31783}%)')
