#%% 
import numpy as np

# Define the vectors
v1 = np.array([1, 2, 2, 3, 4, 6])
v2 = np.array([1, 2, 2, 3, 4, 6])

# List to store the results
results = []
results_uns = []

# Loop over starting positions dynamically
for start in range(len(v1)):  # Iterate through all starting positions
    temp_list = []  # Temporary list for this iteration
    uns =[]
    # Loop to check increasing matches from the current start position
    for i in range(start, len(v1)):  
        if np.array_equal(v1[start:i+1], v2[start:i+1]):  # Compare subarrays
            temp_list.append(i - start + 1)  # Append match length
            uns.append(1)
    results.append(temp_list)  # Store results for this starting index
    results_uns.append(uns)
# %%
# Print the results 
for idx, match in enumerate(results):
    print(f"Starting at index {idx}: {match}")

for idx, match in enumerate(results_uns):
    print(f"Starting at index {idx}: {match}")

#%%
soma =0
for i in results:
    soma += np.sum(i)
    print(i)
    print(soma)
print(soma)
# %%
v2 = np.array([1, 2, 2, 3, 0])
soma = np.sum(v2)
print(soma)
# %%
import mne 
