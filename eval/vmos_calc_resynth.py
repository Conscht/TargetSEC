from wvmos import get_wvmos

model = get_wvmos(cuda=True)
model.eval()

wav_dir = "eval_outputs/resynth_epoch48/wav"
mos = model.calculate_dir(wav_dir, mean=True)
print(f"Resynthesis WVMOS: {mos:.4f}")
