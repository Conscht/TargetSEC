from wvmos import get_wvmos

model = get_wvmos(cuda=True)
model.eval()

wav_root = "eval_outputs/ldm_finetune_epoch596/wav"

total = 0.0
for num in range(1, 8):
    mos = model.calculate_dir(f"{wav_root}/class_{num}", mean=True)
    print(f"Class {num}: WVMOS = {mos:.4f}")
    total += mos

print(f"\nMean WVMOS across all classes: {total/7:.4f}")
