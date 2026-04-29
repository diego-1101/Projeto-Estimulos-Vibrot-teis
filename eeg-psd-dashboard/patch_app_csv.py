import sys

def patch_app_csv():
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # Change in run_performance_analysis
    content = content.replace(
        "csv_path = os.path.join(base_path, 'data', f'analise_df_{prot}_final.csv')",
        "csv_path = os.path.join(base_path, 'data', f'df_prot{prot}_performance.csv')"
    )
    
    # The above replace might catch both instances if they are identical.
    # Let's verify if they are identical in both functions.
    # In run_performance_analysis:
    #         csv_path = os.path.join(base_path, 'data', f'analise_df_{prot}_final.csv')
    # In update_posthoc_plot:
    #     csv_path = os.path.join(base_path, 'data', f'analise_df_{prot}_final.csv')
    # Wait, the spacing might be different. 
    # Let's use a regex-based replace or just replace the substring.
    
    content = content.replace("analise_df_{prot}_final.csv", "df_prot{prot}_performance.csv")

    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    patch_app_csv()
    print("App patched to use performance CSVs.")
