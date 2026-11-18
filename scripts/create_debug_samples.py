"""Synthetic fixtures only; these are not comic accuracy benchmarks."""
from pathlib import Path
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parents[1] / 'evaluation' / 'synthetic'
root.mkdir(parents=True, exist_ok=True)
image = Image.new('RGB', (800, 1200), 'white')
draw = ImageDraw.Draw(image)
for box in [(40, 40, 760, 370), (40, 410, 370, 1150), (410, 410, 760, 1150)]:
    draw.rectangle(box, outline='black', width=8)
image.save(root/'三格测试.png')
Image.new('RGB', (400, 600), 'white').save(root/'空白.png')
(root/'损坏.png').write_bytes(b'intentionally corrupted fixture')
print(root)
