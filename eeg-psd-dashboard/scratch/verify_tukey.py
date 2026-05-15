import numpy as np
import pandas as pd
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.libqsturng import psturng

# Balanced data
n = 10
k = 3
df_resid = (n * k) - k
mse = 1.5

# Generate means
means = [10, 12, 11]
data = []
for i, m in enumerate(means):
    # exact means, variance = mse
    y = np.random.normal(m, np.sqrt(mse), n)
    # adjust to exact mean and mse for testing
    y = (y - np.mean(y)) / np.std(y, ddof=1) * np.sqrt(mse) + m
    for val in y:
        data.append({'val': val, 'group': f'G{i+1}'})

df = pd.DataFrame(data)

# 1. Statsmodels
mc = MultiComparison(df['val'], df['group'])
res = mc.tukeyhsd()
print("Statsmodels Tukey Table:")
print(res)

# 2. Manual
se = np.sqrt(mse / n)
diff = 12 - 10 # G1 vs G2
q = abs(diff) / se
p = psturng(q, k, df_resid)
print(f"\nManual q: {q:.4f}")
print(f"Manual p: {p[0]:.4f}")
