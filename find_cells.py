import json

with open('notebooks/03_feature_engineering.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

target_ids = ['2d67bcdd', 'be6f8c1e', '983ec2d2', '147d0f69', '95958010']

print("Key cells to execute:")
for i, c in enumerate(nb['cells']):
    cid = c.get('id', '')
    if cid in target_ids:
        src = ''.join(c.get('source', []))
        first_line = src.split('\n')[0][:70]
        has_output = len(c.get('outputs', [])) > 0
        status = 'EXECUTED' if has_output else 'NOT RUN'
        print(f"Cell {i}: {first_line}... [{status}]")
