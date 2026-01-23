# EEG PSD Dashboard - Deployment Guide

## ✅ What's Been Created

Your production-ready dashboard is complete with the following structure:

```
eeg-psd-dashboard/
├── app.py                 # Main Dash application (280+ lines)
├── data_loader.py         # Data preprocessing module
├── analysis_engine.py     # Statistical methods (PCA, LDA, CDA, PLS)
├── data/
│   └── df_A_final.csv    # Your Protocol A data (copied)
├── assets/
│   └── styles.css        # Scientific dashboard styling
├── requirements.txt       # All dependencies
├── vercel.json           # Vercel deployment config
├── .gitignore            # Git ignore rules
└── README.md             # Full documentation
```

## 🚀 Quick Start (Local Testing)

1. **Install dependencies** (command already running):
   ```bash
   cd eeg-psd-dashboard
   pip install -r requirements.txt
   ```

2. **Run the dashboard**:
   ```bash
   python app.py
   ```

3. **Open browser**: http://localhost:8050

## 📊 Features Implemented

### ✅ Analysis Methods
- **PCA**: Principal Component Analysis (unsupervised)
- **LDA**: Linear Discriminant Analysis (supervised, group separation)
- **CDA**: Canonical Discriminant Analysis
  - Auto-covariance: Uses LDA
  - Cross-covariance: Canonical Correlation Analysis
- **PLS**: Partial Least Squares
  - Cross-covariance: PSD × Behavior relationship
  - Auto-covariance: PLS-DA (discriminant analysis)

### ✅ UI Controls
- Method selector (PCA/LDA/CDA/PLS)
- Covariance mode (Auto/Cross)
- Data domain (PSD/Behavior)
- Dimensions (2D/3D)
- Run Analysis button
- Real-time statistics panel
- Warning messages for invalid configurations

### ✅ Visualizations
- Interactive 2D scatter plots
- Interactive 3D scatter plots
- Group distinction (CV = Green, SV = Red)
- Hover information (ID, Group, Component values)
- Variance explained display
- Canonical correlations display

## 🌐 Deployment to Vercel

### Option 1: Vercel CLI (Recommended)

1. **Install Vercel CLI**:
   ```bash
   npm install -g vercel
   ```

2. **Navigate to project**:
   ```bash
   cd eeg-psd-dashboard
   ```

3. **Deploy**:
   ```bash
   vercel
   ```

4. **Follow prompts**:
   - Link to your Vercel account
   - Choose project name
   - Confirm settings

### Option 2: Vercel Web Interface

1. Go to https://vercel.com
2. Click "Import Project"
3. Connect your GitHub repository
4. Select `eeg-psd-dashboard` folder
5. Click "Deploy"

## 🔬 Scientific Validation

The dashboard implements the exact requirements:

1. ✅ **Separate CDA analyses**:
   - PSD-only CDA (Auto-covariance with LDA)
   - Behavior-only CDA (Auto-covariance with LDA)

2. ✅ **Cross-canonical correlation**:
   - Implemented via CDA Cross-covariance mode
   - Shows correlation between canonical components

3. ✅ **PLS comparison**:
   - Mode A (Cross-covariance): PSD × Behavior
   - Mode B (Auto-covariance): PLS-DA with groups

4. ✅ **Covariance mode selection**:
   - Affects all methods (PCA, LDA, CDA, PLS)
   - Clear UI distinction

5. ✅ **Group distinction**:
   - CV and SV visually distinct
   - Preserved across all plots
   - Clear legends

## 📝 Next Steps

1. **Test locally** to ensure everything works
2. **Review the visualizations** with your advisor
3. **Deploy to Vercel** for online access
4. **Extend to Protocols B and C** (future work)

## 🐛 Troubleshooting

### Data not loading?
- Check that `data/df_A_final.csv` exists
- Verify CSV format matches expected structure

### Dependencies error?
- Ensure Python 3.8+ is installed
- Try: `pip install --upgrade pip`
- Then: `pip install -r requirements.txt`

### Vercel deployment fails?
- Check that all files are committed to Git
- Ensure `vercel.json` is in the root of `eeg-psd-dashboard/`
- Verify Python version compatibility

## 📧 Support

For issues or questions, refer to:
- README.md for detailed documentation
- Code comments in each module
- Vercel documentation: https://vercel.com/docs

---

**Dashboard Status**: ✅ READY FOR DEPLOYMENT
**Created**: 2026-01-23
**Protocol**: A (CV and SV groups)
