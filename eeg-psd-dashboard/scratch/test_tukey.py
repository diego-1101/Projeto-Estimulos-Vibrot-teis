import numpy as np
from scipy.stats import t
from statsmodels.stats.libqsturng import psturng

# Parameters
df = 20
k = 2
t_val = 2.093 # critical t for p=0.05, df=20 (approx)

# Standard Tukey q for k=2 is t * sqrt(2)
q_val = t_val * np.sqrt(2)

p_tukey = psturng(q_val, k, df)
p_t = 2 * (1 - t.cdf(t_val, df))

print(f"T-test p-value: {p_t:.4f}")
print(f"P_TUKEY TYPE: {type(p_tukey)}")
print(f"P_TUKEY: {p_tukey}")
