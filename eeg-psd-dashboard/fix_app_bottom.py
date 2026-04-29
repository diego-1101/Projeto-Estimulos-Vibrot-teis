import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove the block
block = "if __name__ == '__main__':\n    app.run_server(debug=True, host='0.0.0.0', port=8050)"
content = content.replace(block, "")

# Append to the end
content += "\n\n" + block + "\n"

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
