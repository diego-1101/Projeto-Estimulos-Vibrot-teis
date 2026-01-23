# Dashboard v2.0 - New Features Guide

## ✅ What's New

### 1. **Theme Toggle** 🌙☀️
- **Button location**: Top-right corner of the screen
- **Light mode** (default): Clean white background, dark text
- **Dark mode**: Dark background (#1a1a1a), light text
- **Toggle**: Click the sun/moon button to switch
- **Smooth transitions**: All colors animate smoothly

### 2. **Fixed Dropdown Text Colors** ✅
- **Light mode**: Dark text on white background (readable!)
- **Dark mode**: Light text on dark background (readable!)
- No more dark-on-dark text issue

### 3. **New Comparison Mode** 🔬
- **Purpose**: Compare different analysis methods on the SAME protocol
- **How it works**:
  1. Check "Enable Side-by-Side Comparison"
  2. Left panel shows "Analysis 1" controls
  3. Right panel shows "Analysis 2" controls (appears when comparison is enabled)
  4. Each side has independent settings:
     - Method (PCA, LDA, CDA, PLS)
     - Covariance mode (Auto vs Cross)
     - Data domain (PSD vs Behavior)
     - Dimensions (2D vs 3D)
  5. Click "Run Analysis" to generate both plots

## 🎯 Use Cases

### Example 1: Compare Auto vs Cross Covariance
1. Select **Protocol A**
2. Enable **Comparison Mode**
3. **Left panel**:
   - Method: PLS
   - Covariance: Auto-covariance
   - Domain: PSD
4. **Right panel**:
   - Method: PLS
   - Covariance: Cross-covariance
   - (Domain auto-selected)
5. Click **Run Analysis**
6. See both results side-by-side!

### Example 2: Compare PCA vs LDA
1. Select **Protocol B**
2. Enable **Comparison Mode**
3. **Left panel**: PCA, Auto, PSD, 2D
4. **Right panel**: LDA, Auto, PSD, 2D
5. Click **Run Analysis**
6. Compare unsupervised vs supervised methods!

### Example 3: Compare 2D vs 3D
1. Select **Protocol A**
2. Enable **Comparison Mode**
3. **Left panel**: PCA, Auto, PSD, **2D**
4. **Right panel**: PCA, Auto, PSD, **3D**
5. Click **Run Analysis**
6. See the same data in different dimensions!

## 🎨 Theme Toggle Details

### Light Mode (Default)
- Background: `#f8f9fa` (light gray)
- Cards: `#ffffff` (white)
- Text: `#2c3e50` (dark blue-gray)
- Accent: `#007bff` (blue)
- Dropdowns: White background, dark text

### Dark Mode
- Background: `#1a1a1a` (very dark gray)
- Cards: `#2d2d2d` (dark gray)
- Text: `#e0e0e0` (light gray)
- Accent: `#4fc3f7` (cyan)
- Dropdowns: Dark background (#404040), light text

### Toggle Button
- **Position**: Fixed top-right corner
- **Icon**: 
  - 🌙 (moon) when in light mode → click to go dark
  - ☀️ (sun) when in dark mode → click to go light
- **Hover effect**: Scales up slightly
- **Smooth**: All transitions are 0.3s

## 🔄 How to Update

1. **If server is running**: Just refresh your browser (F5)
2. **If server stopped**: Restart with `python app.py`
3. **Hard refresh**: Ctrl+Shift+R (clears cache)

## 📊 Interface Layout

### Single View Mode (Default)
```
┌─────────────────────────────────────────┐
│  Sidebar          │  Main Plot          │
│  - Protocol       │  [Large Graph]      │
│  - Comparison: □  │                     │
│  - Method         │  Statistics Panel   │
│  - Covariance     │                     │
│  - Domain         │                     │
│  - Dimensions     │                     │
│  [Run Analysis]   │                     │
└─────────────────────────────────────────┘
```

### Comparison Mode (When Enabled)
```
┌───────────────────────────────────────────────────────┐
│  Sidebar              │  Left Plot    │  Right Plot   │
│  - Protocol           │  Analysis 1   │  Analysis 2   │
│  - Comparison: ☑      │  [Graph 1]    │  [Graph 2]    │
│  ┌─────────────────┐  │               │               │
│  │ Analysis 1      │  │  Stats 1      │  Stats 2      │
│  │ - Method        │  │               │               │
│  │ - Covariance    │  │               │               │
│  │ - Domain        │  │               │               │
│  │ - Dimensions    │  │               │               │
│  └─────────────────┘  │               │               │
│  ┌─────────────────┐  │               │               │
│  │ Analysis 2      │  │               │               │
│  │ - Method        │  │               │               │
│  │ - Covariance    │  │               │               │
│  │ - Domain        │  │               │               │
│  │ - Dimensions    │  │               │               │
│  └─────────────────┘  │               │               │
│  [Run Analysis]       │               │               │
└───────────────────────────────────────────────────────┘
```

## 🐛 Troubleshooting

### Dropdown text still not visible?
- Hard refresh: Ctrl+Shift+R
- Clear browser cache
- Restart server

### Theme toggle not working?
- Check browser console for errors (F12)
- Ensure JavaScript is enabled
- Try hard refresh

### Comparison mode not showing?
- Check the "Enable Side-by-Side Comparison" checkbox
- Refresh the page
- Check that both Analysis 1 and Analysis 2 controls appear in sidebar

### Plots not updating?
- Click "Run Analysis" button
- Check that protocol is selected
- Verify data files exist in `data/` folder

## 📝 Technical Details

### Files Modified
1. `app.py` - Complete rewrite with:
   - Theme toggle functionality
   - Comparison mode with independent controls
   - Clientside callback for theme switching
   - Pattern-matching callbacks for dynamic controls

2. `assets/styles.css` - Complete rewrite with:
   - Light and dark theme definitions
   - Proper dropdown text colors for both themes
   - Smooth transitions
   - Comparison panel styling

### Key Features
- **Clientside callback**: Theme switching happens in browser (instant!)
- **Pattern-matching callbacks**: Dynamic controls for comparison panels
- **State management**: Theme persisted in dcc.Store
- **Responsive**: Works on different screen sizes

---

**Version**: 2.0
**Date**: 2026-01-23
**Status**: ✅ READY TO USE
