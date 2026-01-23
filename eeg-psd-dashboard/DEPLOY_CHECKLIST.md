# 🚀 Vercel Deployment Checklist

## ✅ Files Ready for Deployment

I've prepared everything you need:

1. ✅ `app.py` - Added `server = app.server` for Vercel
2. ✅ `vercel.json` - Configured for Dash deployment
3. ✅ `requirements.txt` - All dependencies listed
4. ✅ `data/df_A_final.csv` - Protocol A data
5. ✅ `data/df_B_final.csv` - Protocol B data
6. ✅ `assets/styles.css` - Styling with dark/light themes
7. ✅ `assets/theme.js` - Theme toggle JavaScript
8. ✅ `data_loader.py` - Data processing module
9. ✅ `analysis_engine.py` - Statistical methods

---

## 📝 Step 1: Commit and Push to GitHub

Open terminal in your project root:

```bash
cd "C:\Users\diego\OneDrive\Documents\GitHub\Projeto-Estimulos-Vibrot-teis"

# Check status
git status

# Add all changes
git add .

# Commit
git commit -m "Prepare EEG PSD Dashboard for Vercel deployment"

# Push to GitHub
git push
```

---

## 🌐 Step 2: Deploy on Vercel

### A. Import Repository
1. Go to Vercel dashboard
2. Click **"Import"** next to **"Projeto-Estimulos-Vibrot-teis"**

### B. Configure Project

**CRITICAL SETTINGS:**

| Setting | Value | Why |
|---------|-------|-----|
| **Project Name** | `eeg-psd-dashboard` | Your choice |
| **Framework Preset** | `Other` | Dash is not a standard framework |
| **Root Directory** | `eeg-psd-dashboard` | ⚠️ **MOST IMPORTANT!** |
| **Build Command** | (leave empty) | Dash doesn't need build |
| **Output Directory** | (leave empty) | No static output |
| **Install Command** | (auto-detected) | Uses requirements.txt |

### C. Root Directory Configuration

**This is the KEY step:**

1. Find "Root Directory" setting
2. Click **"Edit"**
3. Type: `eeg-psd-dashboard`
4. Click **"Save"**

This tells Vercel: "Deploy only the dashboard folder, not the entire repo"

### D. Deploy

Click the big **"Deploy"** button!

---

## ⏱️ What Happens Next

Vercel will:
1. ✅ Clone your GitHub repo
2. ✅ Navigate to `eeg-psd-dashboard` folder
3. ✅ Install Python dependencies
4. ✅ Start your Dash app
5. ✅ Give you a live URL!

**Deployment time**: ~2-3 minutes

---

## 🎉 After Deployment

You'll get a URL like:
```
https://eeg-psd-dashboard.vercel.app
```

or

```
https://eeg-psd-dashboard-diego1101.vercel.app
```

**Test it:**
1. Open the URL
2. Check theme toggle works
3. Try running an analysis
4. Test comparison mode

---

## 🐛 If Deployment Fails

### Check Build Logs
1. Click on the failed deployment
2. Read the error message
3. Common issues below:

### Common Issue 1: "Root directory not found"
**Fix**: Make sure you set Root Directory to `eeg-psd-dashboard`

### Common Issue 2: "requirements.txt not found"
**Fix**: 
```bash
# Make sure file exists
ls eeg-psd-dashboard/requirements.txt

# If missing, it should be there - check git
git status
```

### Common Issue 3: "Module 'app' has no attribute 'server'"
**Fix**: Make sure `server = app.server` is in `app.py` (I already added it!)

### Common Issue 4: "Data files not found"
**Fix**: Make sure data files are committed:
```bash
git add eeg-psd-dashboard/data/*.csv
git commit -m "Add data files"
git push
```

---

## 🔄 Future Updates

To update your live dashboard:

1. **Make changes locally**
2. **Test locally** (`python app.py`)
3. **Commit and push**:
   ```bash
   git add .
   git commit -m "Update dashboard"
   git push
   ```
4. **Vercel auto-deploys!** 🎉

No need to manually redeploy - Vercel watches your GitHub repo!

---

## 📊 Vercel Dashboard Features

After deployment, you can:
- ✅ View deployment logs
- ✅ See analytics (visitors, requests)
- ✅ Set custom domain (optional)
- ✅ View previous deployments
- ✅ Rollback if needed

---

## ✅ Final Checklist

Before clicking "Deploy":

- [ ] Committed all changes to Git
- [ ] Pushed to GitHub
- [ ] Set Root Directory to `eeg-psd-dashboard`
- [ ] Selected "Other" as Framework
- [ ] Ready to click "Deploy"!

---

## 🎯 Quick Reference

**Repository**: `Projeto-Estimulos-Vibrot-teis`
**Dashboard Folder**: `eeg-psd-dashboard`
**Main File**: `app.py`
**Entry Point**: `server` variable (already configured!)

---

**Ready to deploy? Follow the steps above!** 🚀

Need help? Check the error logs in Vercel and let me know!
