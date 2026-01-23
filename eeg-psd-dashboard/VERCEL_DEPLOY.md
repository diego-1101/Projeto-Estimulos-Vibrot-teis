# Vercel Deployment Guide - Step by Step

## 📋 Pre-Deployment Checklist

Before deploying, make sure you have:
- ✅ Dashboard working locally
- ✅ GitHub repository connected to Vercel
- ✅ All files committed and pushed to GitHub

---

## 🚀 Step-by-Step Deployment

### Step 1: Import Your Repository

1. On Vercel, click **"Import"** next to **"Projeto-Estimulos-Vibrot-teis"**
2. Vercel will ask you to configure the project

### Step 2: Configure Project Settings

**IMPORTANT**: Since your dashboard is in a subdirectory, you need to configure this:

#### **Root Directory**
- Click on **"Edit"** next to "Root Directory"
- Enter: `eeg-psd-dashboard`
- This tells Vercel to deploy only the dashboard folder

#### **Framework Preset**
- Select: **"Other"** (not Next.js, not React, just "Other")

#### **Build Command**
- Leave empty or enter: `pip install -r requirements.txt`

#### **Output Directory**
- Leave empty (Dash doesn't need a build step)

#### **Install Command**
- Leave as default: `pip install -r requirements.txt`

### Step 3: Environment Variables (Optional)

If you have any secrets (you don't for now), you'd add them here.
- Skip this for now

### Step 4: Deploy!

Click **"Deploy"**

Vercel will:
1. Clone your repository
2. Navigate to `eeg-psd-dashboard` folder
3. Install dependencies from `requirements.txt`
4. Start your Dash app

---

## ⚠️ IMPORTANT: Update `vercel.json`

Before deploying, you need to update your `vercel.json` file because Dash apps need a special configuration:

### Current `vercel.json` (needs update):
```json
{
  "version": 2,
  "builds": [
    {
      "src": "app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "app.py"
    }
  ]
}
```

### **NEW `vercel.json`** (for Dash apps):
```json
{
  "version": 2,
  "builds": [
    {
      "src": "app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "app.py"
    }
  ],
  "env": {
    "PYTHON_VERSION": "3.9"
  }
}
```

---

## 📝 Update `app.py` for Vercel

Vercel needs a special server configuration. Update the last lines of `app.py`:

### Current (for local):
```python
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
```

### **NEW (for Vercel)**:
```python
# For Vercel deployment
server = app.server

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
```

The `server = app.server` line is what Vercel needs!

---

## 🔧 Complete Pre-Deployment Steps

### 1. Update `vercel.json`
I'll create the correct version for you.

### 2. Update `app.py`
Add the `server = app.server` line.

### 3. Commit and Push to GitHub
```bash
cd "C:\Users\diego\OneDrive\Documents\GitHub\Projeto-Estimulos-Vibrot-teis"
git add .
git commit -m "Prepare dashboard for Vercel deployment"
git push
```

### 4. Deploy on Vercel
- Click "Import" on your repository
- Set Root Directory: `eeg-psd-dashboard`
- Click "Deploy"

---

## 🎯 Vercel Configuration Summary

| Setting | Value |
|---------|-------|
| **Root Directory** | `eeg-psd-dashboard` |
| **Framework** | Other |
| **Build Command** | (empty) |
| **Output Directory** | (empty) |
| **Install Command** | `pip install -r requirements.txt` |

---

## 🐛 Common Issues & Solutions

### Issue 1: "Module not found"
**Solution**: Make sure `requirements.txt` is in the `eeg-psd-dashboard` folder

### Issue 2: "Application failed to start"
**Solution**: Check that `server = app.server` is in `app.py`

### Issue 3: "Build failed"
**Solution**: Check Vercel build logs for specific error

### Issue 4: "Data files not found"
**Solution**: Make sure `data/df_A_final.csv` and `data/df_B_final.csv` are committed to Git

---

## 📊 After Deployment

Once deployed, Vercel will give you a URL like:
```
https://your-project-name.vercel.app
```

The dashboard will be live at that URL!

---

## 🔄 Future Updates

To update your deployed dashboard:
1. Make changes locally
2. Test locally
3. Commit and push to GitHub
4. Vercel automatically redeploys! 🎉

---

## ✅ Next Steps

1. I'll update the necessary files
2. You commit and push to GitHub
3. You import on Vercel with correct settings
4. Dashboard goes live!

Ready to proceed?
