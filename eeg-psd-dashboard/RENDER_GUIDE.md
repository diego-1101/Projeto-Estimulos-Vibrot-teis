# 🚀 Redeploying to Render

Since we've updated the dashboard with new features, here is how to deploy it to Render.

## 1. Commit and Push Changes to GitHub

The most important step is to send your local changes to GitHub. Render watches your repository and will (usually) automatically deploy when it sees new code.

Run these commands in your terminal:

```bash
cd "C:\Users\diego\OneDrive\Documents\GitHub\Projeto-Estimulos-Vibrot-teis"

# Add all new files (including data)
git add .

# Commit with a message
git commit -m "Add Complexity and Overlap visualization features"

# Push to GitHub
git push origin main
```

## 2. Check Render Dashboard

1.  Go to [dashboard.render.com](https://dashboard.render.com/).
2.  Click on your **eeg-psd-dashboard** service.
3.  Click **Events** or **Logs** in the sidebar.
4.  You should see a new deployment starting with the message "Add Complexity and Overlap visualization features".

**If it started automatically:**
- Just wait for it to finish! (It takes 3-5 minutes).

**If it DID NOT start automatically:**
1.  Click the blue **Manual Deploy** button (top right).
2.  Select **Deploy latest commit**.

## 3. Verify New Features

Once deployed (status changes to "Live"):
1.  Open your dashboard URL (e.g., `https://eeg-psd-dashboard.onrender.com`).
2.  Select **Protocol A**.
3.  Look for the new **"Color By"** dropdown in the Analysis panel.
4.  Try selecting **"Complexity"** or **"Overlap"**.
5.  Check if the points change color and the legend updates.

## ⚠️ Common Issues

- **Data Missing on Render**: If the graph works but "Color By" doesn't change anything, it might mean the new CSV files (`complexidade_protA.csv`, etc.) weren't uploaded.
    - *Fix*: Ensure you ran `git add .` and `git push`.
- **Build Fail**: Check the logs. If it says "Module not found", we might need to update requirements (but we didn't add new libraries, so this is unlikely).
