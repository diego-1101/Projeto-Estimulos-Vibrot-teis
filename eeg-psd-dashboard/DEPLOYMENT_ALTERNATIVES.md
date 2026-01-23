# ⚠️ Vercel Deployment Size Issue - Solutions

## The Problem

**Error**: "A Serverless Function has exceeded the unzipped maximum size of 250 MB"

**Cause**: Scientific Python libraries (`numpy`, `pandas`, `scikit-learn`) are very large:
- numpy: ~50 MB
- pandas: ~40 MB
- scikit-learn: ~80 MB
- Total with dependencies: ~250+ MB

Vercel's free tier limit: **250 MB** for serverless functions.

---

## 🚀 Solution Options

### **Option 1: Use Render.com (RECOMMENDED)** ✅

Render is better for Python apps with heavy dependencies.

#### Steps:
1. Go to https://render.com
2. Sign up with GitHub
3. Click "New +" → "Web Service"
4. Connect your repository
5. Configure:
   - **Root Directory**: `eeg-psd-dashboard`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:server`
   - **Environment**: Python 3
6. Click "Create Web Service"

**Advantages**:
- ✅ No size limit
- ✅ Free tier available
- ✅ Better for scientific Python
- ✅ Automatic HTTPS
- ✅ Custom domains

**Disadvantages**:
- ⚠️ Slower cold starts (30s-1min)
- ⚠️ Free tier sleeps after 15min inactivity

---

### **Option 2: Use Railway.app** ✅

Similar to Render, good for Python apps.

#### Steps:
1. Go to https://railway.app
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Set root directory: `eeg-psd-dashboard`
6. Railway auto-detects Python and deploys

**Advantages**:
- ✅ Very easy setup
- ✅ No size limit
- ✅ Fast deployments
- ✅ $5 free credit monthly

---

### **Option 3: Optimize for Vercel** ⚠️

Try to reduce the package size (difficult).

#### Create `vercel.json` with optimization:
```json
{
  "version": 2,
  "builds": [
    {
      "src": "app.py",
      "use": "@vercel/python",
      "config": {
        "maxLambdaSize": "50mb"
      }
    }
  ],
  "functions": {
    "app.py": {
      "memory": 3008,
      "maxDuration": 60
    }
  }
}
```

#### Reduce `requirements.txt`:
```
dash==2.17.0
plotly==5.18.0
pandas==2.1.0
numpy==1.24.0
scikit-learn==1.3.0
```

**This might not work** - the libraries are still too large.

---

### **Option 4: Use PythonAnywhere** 💰

Paid option but very reliable.

1. Go to https://www.pythonanywhere.com
2. Sign up (free tier available but limited)
3. Upload your code
4. Configure WSGI
5. Deploy

**Cost**: ~$5/month for basic plan

---

### **Option 5: Use Heroku** 💰

Classic platform, now paid only.

**Cost**: ~$5-7/month

---

## 🎯 My Recommendation

### **Use Render.com** (Free)

**Why**:
1. ✅ **Free tier** with no size limits
2. ✅ **Easy deployment** (similar to Vercel)
3. ✅ **Perfect for Python** scientific apps
4. ✅ **Automatic deployments** from GitHub
5. ✅ **Custom domains** supported

**Only downside**: Cold starts (first load takes 30s-1min after inactivity)

---

## 📝 Quick Render Deployment Guide

### Step 1: Create `Procfile`

Create a file named `Procfile` (no extension) in `eeg-psd-dashboard/`:

```
web: gunicorn app:server --bind 0.0.0.0:$PORT
```

### Step 2: Update `requirements.txt`

Make sure it has gunicorn:
```
dash==2.18.1
plotly==5.24.1
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.5.2
gunicorn==23.0.0
```

### Step 3: Deploy on Render

1. Go to https://render.com
2. Sign up with GitHub
3. Click "New +" → "Web Service"
4. Select "Projeto-Estimulos-Vibrot-teis"
5. Configure:
   - Name: `eeg-psd-dashboard`
   - Root Directory: `eeg-psd-dashboard`
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:server --bind 0.0.0.0:$PORT`
6. Click "Create Web Service"

**Done!** Your app will be live at: `https://eeg-psd-dashboard.onrender.com`

---

## ⚡ Alternative: Keep Trying Vercel

If you really want Vercel, you can:

1. **Upgrade to Pro** ($20/month) - increases limit to 50MB per function
2. **Split the app** - separate frontend and backend (complex)
3. **Use Docker** - Vercel supports Docker (more setup)

But honestly, **Render is easier and free**! 🎉

---

## 🔄 Next Steps

1. **Choose a platform** (I recommend Render)
2. **I'll help you deploy** there
3. **Get your dashboard live!**

Which option do you prefer?
