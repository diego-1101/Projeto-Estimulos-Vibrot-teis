# EEG PSD Dashboard - Protocol A

Interactive scientific dashboard for analyzing EEG Power Spectral Density (PSD) features and behavioral performance data from Protocol A.

## Features

- **Multiple Analysis Methods**: PCA, LDA, CDA, PLS
- **Covariance Modes**: Auto-covariance (X with X) and Cross-covariance (PSD × Behavior)
- **Interactive Visualizations**: 2D and 3D scatter plots with group distinction (CV vs SV)
- **Real-time Statistics**: Variance explained, canonical correlations

## Local Development

### Installation

```bash
pip install -r requirements.txt
```

### Running the App

```bash
python app.py
```

The dashboard will be available at `http://localhost:8050`

## Deployment to Vercel

1. Install Vercel CLI:
```bash
npm install -g vercel
```

2. Deploy:
```bash
vercel
```

3. Follow the prompts to link your project.

## Data Structure

The dashboard uses `data/df_A_final.csv` which contains:
- **PSD Features**: Normalized spectral power bands (delta, theta, alpha, beta, gamma) for channels CZ, C3, C4
- **Behavioral Metrics**: Desempenho, Acuracia, Similaridade, Especificidade
- **Group Labels**: CV (Com Vibração) and SV (Sem Vibração)

## Usage

1. **Select Analysis Method**: Choose between PCA, LDA, CDA, or PLS
2. **Choose Covariance Mode**: 
   - Auto-covariance: Analyzes single domain (PSD or Behavior)
   - Cross-covariance: Analyzes relationship between PSD and Behavior
3. **Select Data Domain**: PSD features or Behavioral metrics (disabled for cross-covariance)
4. **Choose Dimensions**: 2D or 3D visualization
5. **Click "Run Analysis"**: Generate the embedding and view results

## Scientific Methods

### PCA (Principal Component Analysis)
Unsupervised dimensionality reduction maximizing variance.

### LDA (Linear Discriminant Analysis)
Supervised method maximizing group separation (CV vs SV).

### CDA (Canonical Discriminant Analysis)
- **Auto-covariance**: Same as LDA
- **Cross-covariance**: Canonical Correlation Analysis between PSD and Behavior

### PLS (Partial Least Squares)
- **Cross-covariance**: Finds latent variables maximizing covariance between PSD and Behavior
- **Auto-covariance (PLS-DA)**: Discriminant analysis using group labels

## Project Structure

```
eeg-psd-dashboard/
├── app.py                 # Main Dash application
├── data_loader.py         # Data loading and preprocessing
├── analysis_engine.py     # Statistical methods implementation
├── data/
│   └── df_A_final.csv    # Protocol A dataset
├── assets/
│   └── styles.css        # Custom styling
├── requirements.txt       # Python dependencies
├── vercel.json           # Vercel deployment config
└── README.md             # This file
```

## Author

Diego - Projeto Estímulos Vibrotáteis
