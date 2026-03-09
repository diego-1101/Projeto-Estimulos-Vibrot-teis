import requests
import json
import re

payload = {
    'output': '..plot-1.figure...stats-1.children...anova-plot-1.figure...anova-stats-1.children...info-panel.children..',
    'outputs': {'id': '..plot-1.figure...stats-1.children...anova-plot-1.figure...anova-stats-1.children...info-panel.children..', 'property': '', 'timestamp': 1700000000},
    'inputs': [{'id': 'run-btn', 'property': 'n_clicks', 'value': 1}],
    'changedPropIds': ['run-btn.n_clicks'],
    'state': [
        {'id': 'protocol-dropdown', 'property': 'value', 'value': 'A'},
        {'id': {'index': 1, 'type': 'method-dropdown'}, 'property': 'value', 'value': 'PCA'},
        {'id': {'index': 1, 'type': 'x-mode-dropdown'}, 'property': 'value', 'value': 'psd_bands_norm'},
        {'id': {'index': 1, 'type': 'y-checklist'}, 'property': 'value', 'value': ['Desempenho']},
        {'id': {'index': 1, 'type': 'domain-dropdown'}, 'property': 'value', 'value': 'x'},
        {'id': 'global-dimensions-radio', 'property': 'value', 'value': '3'},
        {'id': {'index': 1, 'type': 'color-dropdown'}, 'property': 'value', 'value': 'group'},
        {'id': {'index': 1, 'type': 'anova-target'}, 'property': 'value', 'value': 'Desempenho'},
        {'id': 'theme-store', 'property': 'data', 'value': 'light'},
        {'id': 'comparison-toggle', 'property': 'value', 'value': []},
        {'id': {'axis': 1, 'index': 1, 'type': 'axis-select'}, 'property': 'value', 'value': 'C1_X'},
        {'id': {'axis': 2, 'index': 1, 'type': 'axis-select'}, 'property': 'value', 'value': 'C2_X'},
        {'id': {'axis': 3, 'index': 1, 'type': 'axis-select'}, 'property': 'value', 'value': 'C3_X'}
    ]
}

try:
    r = requests.post('http://localhost:8050/_dash-update-component', json=payload)
    if r.status_code != 200:
        match = re.search(r'(<div class="traceback".*?)</div', r.text, re.DOTALL)
        if match:
            # clean html
            clean = re.sub(r'<[^>]+>', '', match.group(1))
            print("TRACEBACK:")
            print(clean)
        else:
            print(r.text[:2000])
    else:
        print("SUCCESS!")
        print(r.json().keys())
except Exception as e:
    print(e)
