import argparse
from wvmos import get_wvmos

parser = argparse.ArgumentParser()
parser.add_argument("--wav_root", default="eval_outputs/hifigan_mlp_epoch104/wav")
args, _ = parser.parse_known_args()
wav_root = args.wav_root

model = get_wvmos(cuda=True)
model.eval()

total = 0.0
for num in range(1, 8):
    mos = model.calculate_dir(f"{wav_root}/class_{num}", mean=True)
    print(f"Class {num}: WVMOS = {mos:.4f}")
    total += mos

print(f"\nMean WVMOS across all classes: {total/7:.4f}")
