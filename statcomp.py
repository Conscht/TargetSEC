# compare_stats_pt.py
import torch

STATS_FULL  = "style_stats_new.pt"   # from whole audio
STATS_SLICE = "style_stats_k8.pt"  # from random slices

a = torch.load(STATS_FULL,  map_location="cpu")
b = torch.load(STATS_SLICE, map_location="cpu")

mean_a, std_a = a["mean"], a["std"]
mean_b, std_b = b["mean"], b["std"]

print("=== MEAN ===")
print("mean abs diff (avg):", (mean_a - mean_b).abs().mean().item())
print("mean abs diff (max):", (mean_a - mean_b).abs().max().item())

print("\n=== STD ===")
print("std abs diff (avg):", (std_a - std_b).abs().mean().item())
print("std abs diff (max):", (std_a - std_b).abs().max().item())

print("\n=== STD RANGES ===")
print("SLICE  std(min/median/max):",
      std_a.min().item(), std_a.median().item(), std_a.max().item())
print("k8 std(min/median/max):",
      std_b.min().item(), std_b.median().item(), std_b.max().item())

print("\n=== IMPLIED NORMALIZED NORM ===")
print("SLICE  expected ||z|| ~", (std_a.pow(2).sum()).sqrt().item())
print("K8 expected ||z|| ~", (std_b.pow(2).sum()).sqrt().item())
