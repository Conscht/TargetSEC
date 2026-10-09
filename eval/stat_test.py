# import json
# import numpy as np
# from scipy import stats

# # 1. Load your raw data
# # Replace 'results.jsonl' with your actual file path
# sq_errors = []
# with open('results.jsonl', 'r') as f:
#     for line in f:
#         data = json.loads(line)
#         sq_errors.append(data['sq_err'])

# # Convert to numpy array for precision
# sq_errors = np.array(sq_errors)

# # 2. Define the Baseline Mean (mu) from the literature
# baseline_mse = 0.069

# # 3. Perform the One-Sample T-Test
# # We use a one-sided test because our hypothesis is that TargetSEC is BETTER (Lower MSE)
# t_stat, p_value_two_sided = stats.ttest_1samp(sq_errors, baseline_mse)

# # Convert two-sided p-value to one-sided
# # We divide by 2 and check if the t-statistic is negative (meaning our mean is lower)
# if t_stat < 0:
#     p_value_one_sided = p_value_two_sided / 2
# else:
#     p_value_one_sided = 1 - (p_value_two_sided / 2)

# print(f"TargetSEC Mean MSE: {np.mean(sq_errors):.6f}")
# print(f"T-statistic: {t_stat:.4f}")
# print(f"One-sided P-value: {p_value_one_sided:.10f}")

# if p_value_one_sided < 0.05:
#     print("Result is Statistically Significant (p < 0.05)")



import json
import numpy as np
from collections import defaultdict

path = "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM/eval_outputs/test1_MLP_ABLAT_spkFalse/metadata.jsonl"

# collect sq_err per batch_idx
by_batch = defaultdict(list)
with open(path, "r") as f:
    for line in f:
        d = json.loads(line)
        by_batch[d["batch_idx"]].append(d["sq_err"])

# one value per source utterance: mean sq_err across target classes
batch_mse = np.array([np.mean(v) for v in by_batch.values()], dtype=float)

n = len(batch_mse)
mean = batch_mse.mean()
std = batch_mse.std(ddof=1)

# 95% CI (normal approx). For n>30 this is fine; otherwise you can switch to t.
ci_low = mean - 1.96 * std / np.sqrt(n)
ci_high = mean + 1.96 * std / np.sqrt(n)

print(f"N (batch_idx units): {n}")
print(f"TargetSEC MSE: {mean:.6f}")
print(f"95% CI: [{ci_low:.6f}, {ci_high:.6f}]")
