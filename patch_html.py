import re

with open('templates/dashboard.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Add metadata upload field
html = re.sub(
    r'(<div class="form-group">\s*<label for="model-type">Model Type</label>.*?</select>\s*</div>)',
    r'\1\n                            <div class="form-group" id="metadata-group" style="display: none;">\n                                <label>Metadata JSON (for .joblib models)</label>\n                                <div class="drop-zone" id="metadata-drop-zone" onclick="document.getElementById(\'metadata-file\').click()">\n                                    <div>Drop metadata.json here or click to upload</div>\n                                    <div id="metadata-file-name" class="file-name"></div>\n                                </div>\n                                <input type="file" id="metadata-file" accept=".json" style="display: none;">\n                            </div>',
    html, flags=re.DOTALL
)

# Add event listener to show/hide metadata based on model extension
html = re.sub(
    r"(document\.addEventListener\('DOMContentLoaded', \(\) => \{)",
    r"\1\n            document.getElementById('model-file').addEventListener('change', function(e) {\n                if(this.files[0] && this.files[0].name.endsWith('.joblib')) {\n                    document.getElementById('metadata-group').style.display = 'block';\n                } else {\n                    document.getElementById('metadata-group').style.display = 'none';\n                }\n            });",
    html
)

# Add metadata field to FormData
html = re.sub(
    r"(formData\.append\('file', fileInput\.files\[0\]\);)",
    r"\1\n            const metadataFile = document.getElementById('metadata-file').files[0];\n            if (metadataFile) {\n                formData.append('metadata_file', metadataFile);\n            }",
    html
)

# Setup drag drop for metadata
html = re.sub(
    r"(setupDragDrop\('model-drop-zone', 'model-file', 'model-file-name'\);)",
    r"\1\n            setupDragDrop('metadata-drop-zone', 'metadata-file', 'metadata-file-name');",
    html
)

with open('templates/dashboard.html', 'w', encoding='utf-8') as f:
    f.write(html)
print('Dashboard HTML updated')
