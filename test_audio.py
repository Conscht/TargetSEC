from wvmos import get_wvmos
import soundfile as sf
import librosa
import torch
model = get_wvmos(cuda=True)
model.eval()


def calculate_vmos(audio, batchid, Emoclass=None):

    if Emoclass:
        audio = audio.detach().squeeze(1).squeeze(0).cpu().numpy()  # Detach from computation graph
        path = "/data/rajprabhu/dataset/1auga/ba-constantin-ldm/Eval_LDM_/" + str(Emoclass) + "/" + str(batchid) + ".wav"

    else:
        path = "/data/rajprabhu/dataset/1auga/ba-constantin-ldm/Eval/" + str(batchid) + ".wav"

    # Convert the tensor to a NumPy array and make sure it's 1D (mono audio)
    if isinstance(audio, torch.Tensor):
        audio = audio.squeeze(1).squeeze(0).cpu().numpy()  # Convert tensor to NumPy, squeeze to remove any extra dimensions
    print(audio, audio.shape)
    print("EMO CLASS", Emoclass)
    sf.write(path, audio, samplerate=16000)
    mos = model.calculate_one(path) # infer MOS score for one audio 
    return mos

