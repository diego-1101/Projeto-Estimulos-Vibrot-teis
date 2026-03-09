from app import update_single
import traceback

try:
    # def update_single(n, prot, meth, x_mode, y_cols, dom, dims, color, anova_target, theme, comp, ax1, ax2, ax3):
    update_single(1, 'A', 'PCA', 'psd_bands_norm', ['Desempenho'], 'x', 3, 'group', 'Desempenho', 'light', [], 'C1_X', 'C2_X', 'C3_X')
except Exception as e:
    traceback.print_exc()
