from wvmos import get_wvmos
import soundfile as sf
import librosa
import torch
import pathlib

model = get_wvmos(cuda=True)
model.eval()

for num in range(1, 8):
    mos = model.calculate_dir("/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM/eval_outputs/test1_emo_spk_768crossatt_synth_long_final_eval_guidance4__gs07_guidance07/wav/class_" + str(num), mean=True) 
    print(mos)
    print("yeah" + str(num))
# mos = model.calculate_one("/data/rajprabhu/dataset/1auga/ba-constantin-ldm/Eval_LDM_/1/0.wav")
# print(mos)
# mos = model.calculate_one("/data/rajprabhu/dataset/1auga/ba-constantin-ldm/Eval_LDM_/1/1.wav")
# print(mos)
# mos = model.calculate_one("/data/rajprabhu/dataset/1auga/ba-constantin-ldm/Eval_LDM_/1/2.wav")
# print(mos)
# mos = model.calculate_one("/data/rajprabhu/dataset/1auga/ba-constantin-ldm/Eval_LDM_/1/3.wav")
# print(mos)
# mos = model.calculate_one("/data/rajprabhu/dataset/1auga/ba-constantin-ldm/Eval_LDM_/1/4.wav")
# print(mos)
# mos = model.calculate_one("/data/rajprabhu/dataset/1auga/ba-constantin-ldm/Eval_LDM_/1/5.wav")
# print(mos)

# mos = model.calculate_one("/data/rajprabhu/dataset/1auga/ba-constantin-ldm/Eval/0.wav")
# print(mos)
# mos = model.calculate_one("/data/rajprabhu/dataset/1auga/ba-constantin-ldm/Eval/1.wav")
# print(mos)
# mos = model.calculate_one("/data/rajprabhu/dataset/1auga/ba-constantin-ldm/Eval/2.wav")
# print(mos)
# mos = model.calculate_one("/data/rajprabhu/dataset/1auga/ba-constantin-ldm/Eval/3.wav")
# print(mos)
# mos = model.calculate_one("/data/rajprabhu/dataset/1auga/ba-constantin-ldm/Eval/4.wav")
# print(mos)
# mos = model.calculate_one("/data/rajprabhu/dataset/1auga/ba-constantin-ldm/Eval/5.wav")
# print(mos)import os

# import os


# import torch

# def consolidate_embeddings(folder_path, output_file):
#     """
#     Loads all .pt files in a folder, bundles them into a dictionary, and saves as a single .pth file.
    
#     Args:
#         folder_path (str): Path to the folder containing .pt embedding files.
#         output_file (str): Path where the consolidated .pth file will be saved.
#     """
#     all_embeddings = {}
#     file_count = 0

#     for fname in os.listdir(folder_path):
#         if fname.endswith(".pt"):
#             key = fname.replace(".pt", ".wav")  # Use consistent key names
#             full_path = os.path.join(folder_path, fname)
#             try:
#                 embedding = torch.load(full_path, map_location="cpu")
#                 all_embeddings[key] = embedding
#                 file_count += 1
#             except Exception as e:
#                 print(f"Failed to load {fname}: {e}")
    
#     torch.save(all_embeddings, output_file)
#     print(f"[✔] Saved {file_count} embeddings to {output_file}")
# consolidate_embeddings(
#     folder_path="/Users/Conscht/Documents/New folder/emotion_embeddings",
#     output_file="emotion_embeddings.pth"
# )

# consolidate_embeddings(
#     folder_path="/Users/Conscht/Documents/New folder/flat_style_embeddings",
#     output_file="style_embeddings.pth"
# # )
# import torch

# # Replace with your path
# import os
# import torch

# import torch

# pth_path = r"C:\Users\Conscht\Documents\New folder\Audio\MSP-Podcast-1.10\style_embeddings.pth"

# print(f"Loading keys from: {pth_path}")
# emb_dict = torch.load(pth_path, map_location="cpu")

# for key in emb_dict:
#     if "0735" in key:
#         print(f"🧐 Match found: {key}")

# if "MSP-PODCAST_0103_0735.wav" in emb_dict:
#     print("✅ Exact match key found!")
# else:
#     print("❌ Exact match key NOT found.")
