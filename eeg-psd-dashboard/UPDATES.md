# Dashboard Updates - Summary

## ✅ Fixed Issues

### 1. PLSCanonical Error (FIXED)
**Problem**: `'PLSCanonical' object has no attribute 'x_scores_'`
**Solution**: Updated `analysis_engine.py` to use `transform()` method instead of accessing `x_scores_` attribute.

### 2. Dark Mode (IMPLEMENTED)
**Changes**:
- Updated `assets/styles.css` with dark theme
- Background: `#1a1a1a`
- Cards: `#2d2d2d`
- Accent color: `#4fc3f7` (cyan)
- All plots now use `plotly_dark` template

### 3. Protocol B Support (IMPLEMENTED)
**Changes**:
- Updated `data_loader.py` to accept `protocol` parameter
- Copied `df_B_final.csv` to data folder
- Added Protocol dropdown in sidebar (A, B, or Compare)

### 4. Comparison View (IMPLEMENTED)
**New Feature**: Select "Compare A vs B" to see both protocols overlaid
- Different symbols for Protocol A vs B
- Side-by-side statistics
- Same analysis method applied to both

## 📝 How to Update the Dashboard

### Method 1: Automatic Refresh (Development Mode)

When running the dashboard in **debug mode** (which is the default), Dash automatically detects file changes and reloads:

1. **Keep the server running** (`python app.py`)
2. **Save any file changes**
3. **Refresh your browser** (F5 or Ctrl+R)
4. Dash will automatically reload the updated code

**Note**: You'll see this message in the terminal when files change:
```
* Detected change in 'app.py', reloading
* Restarting with stat
```

### Method 2: Manual Restart

If automatic reload doesn't work:

1. **Stop the server**: Press `Ctrl+C` in the terminal
2. **Restart**: Run `python app.py` again
3. **Refresh browser**: Open http://localhost:8050

### Method 3: Hard Refresh (Clear Cache)

If you see old styles or behavior:

1. **Windows/Linux**: `Ctrl + Shift + R` or `Ctrl + F5`
2. **Mac**: `Cmd + Shift + R`

This clears the browser cache and forces a full reload.

## 🎨 New Features Guide

### Protocol Selection

1. **Protocol A**: Original dataset (CV vs SV groups)
2. **Protocol B**: Second dataset (CF vs SF groups)
3. **Compare A vs B**: Overlay both protocols
   - Circles = Protocol A
   - Diamonds = Protocol B
   - Colors still represent groups (Green=CV/CF, Red=SV/SF)

### Comparison Mode

When you select "Compare A vs B":
- Both datasets are analyzed with the same method
- Results are overlaid on the same plot
- Statistics show variance explained for each protocol separately
- Useful for seeing if the same analysis reveals similar patterns

## 🔧 Deployment Updates

### For Vercel Deployment

The changes are already in the files. To deploy:

```bash
cd eeg-psd-dashboard
vercel --prod
```

Vercel will automatically:
1. Detect the changes
2. Rebuild the application
3. Deploy the new version
4. Give you a new URL (or update the existing one)

### For Local Sharing

If you want to share the dashboard on your local network:

```bash
python app.py
```

Then share your IP address: `http://YOUR_IP:8050`

Others on the same network can access it.

## 📊 Testing the Fixes

### Test 1: CDA Cross-Covariance (The Error You Saw)
1. Select **Protocol**: A
2. Select **Method**: CDA
3. Select **Covariance**: Cross-covariance
4. Select **Domain**: Behavioral Metrics (will be auto-selected)
5. Click **Run Analysis**
6. ✅ Should work now without the `x_scores_` error

### Test 2: Dark Mode
1. Open the dashboard
2. ✅ Background should be dark (#1a1a1a)
3. ✅ Sidebar should be dark gray (#2d2d2d)
4. ✅ Plots should have dark background

### Test 3: Protocol B
1. Select **Protocol**: B
2. Select any method
3. Click **Run Analysis**
4. ✅ Should show Protocol B data

### Test 4: Comparison
1. Select **Protocol**: Compare A vs B
2. Select **Method**: PCA
3. Click **Run Analysis**
4. ✅ Should show both protocols with different symbols

## 🐛 Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### Changes not appearing
1. Stop server (Ctrl+C)
2. Clear browser cache (Ctrl+Shift+R)
3. Restart server (`python app.py`)

### Port already in use
```bash
# Kill the process on port 8050
netstat -ano | findstr :8050
taskkill /PID <PID_NUMBER> /F
```

Then restart the server.

## 📁 Modified Files

1. ✅ `analysis_engine.py` - Fixed PLSCanonical
2. ✅ `assets/styles.css` - Dark mode theme
3. ✅ `data_loader.py` - Protocol parameter
4. ✅ `app.py` - Protocol selection + comparison
5. ✅ `data/df_B_final.csv` - Added Protocol B data

## 🚀 Next Steps

1. **Test locally** - Verify all fixes work
2. **Deploy to Vercel** - Share with your advisor
3. **Extend to Protocol C** - Same pattern as B
4. **Add more comparisons** - E.g., compare methods side-by-side

---

**Status**: ✅ ALL ISSUES FIXED AND FEATURES IMPLEMENTED
**Last Updated**: 2026-01-23 01:48
