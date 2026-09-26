import re

with open('templates/dashboard.html', 'r', encoding='utf-8') as f:
    html = f.read()

# 1. Fix the HTML layout for the model upload
html_target = '''<select class="input" id="model-type" required>
                            <option value="">Select Type</option>
                            <option value="MLP">MLP</option>
                            <option value="ANN">ANN</option>
                            <option value="RNN">RNN</option>
                            <option value="LightGBM">LightGBM</option>
                        </select>
                    </div>'''

new_html = '''<select class="input" id="model-type" required onchange="handleModelTypeChange()">
                            <option value="">Select Type</option>
                            <option value="MLP">MLP</option>
                            <option value="ANN">ANN</option>
                            <option value="RNN">RNN</option>
                            <option value="LightGBM">LightGBM</option>
                        </select>
                    </div>
                    
                    <div class="form-group" id="metadata-group" style="display: none; margin-bottom: 16px;">
                        <label style="font-size: 14px; margin-bottom: 8px; display: block;">Metadata JSON (Required for MLP .joblib)</label>
                        <div class="drop-zone" id="metadata-drop-zone" onclick="document.getElementById('metadata-file').click()" style="min-height: 80px; padding: 15px;">
                            <span style="font-size: 20px;">??</span>
                            <p>Drop metadata.json here or click</p>
                            <input type="file" id="metadata-file" accept=".json" style="display: none;">
                        </div>
                        <div id="metadata-file-name" style="margin-top: 8px; font-size: 14px; color: var(--accent);"></div>
                    </div>'''

html = html.replace(html_target, new_html)

# 2. Fix the JavaScript logic in uploadModel()
upload_model_target = '''const formData = new FormData();
            formData.append('model_name', name);
            formData.append('model_type', type);
            formData.append('file', fileInput.files[0]);'''

new_upload_model = '''const formData = new FormData();
            formData.append('model_name', name);
            formData.append('model_type', type);
            formData.append('file', fileInput.files[0]);
            
            if (type === 'MLP') {
                const metadataFile = document.getElementById('metadata-file').files[0];
                if (metadataFile) {
                    formData.append('metadata_file', metadataFile);
                }
            }'''

html = html.replace(upload_model_target, new_upload_model)

# 3. Clean up the messed up uploadDataset() logic
dataset_target = '''const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            const metadataFile = document.getElementById('metadata-file').files[0];
            if (metadataFile) {
                formData.append('metadata_file', metadataFile);
            }'''

new_dataset = '''const formData = new FormData();
            formData.append('file', fileInput.files[0]);'''

html = html.replace(dataset_target, new_dataset)

# 4. Add the handleModelTypeChange function and setupDragDrop for metadata
js_target = '''// --- Models API ---'''
new_js = '''// --- Models API ---
        function handleModelTypeChange() {
            const type = document.getElementById('model-type').value;
            const metaGroup = document.getElementById('metadata-group');
            const modelFile = document.getElementById('model-file');
            
            if (type === 'MLP') {
                metaGroup.style.display = 'block';
                modelFile.accept = '.joblib,.pkl';
                document.getElementById('model-drop-zone').querySelector('p').textContent = 'Drag and drop .joblib file here, or click to browse';
            } else {
                metaGroup.style.display = 'none';
                modelFile.accept = '.h5,.keras';
                document.getElementById('model-drop-zone').querySelector('p').textContent = 'Drag and drop .h5 or .keras file here, or click to browse';
            }
        }
'''
html = html.replace(js_target, new_js)

# Add drag and drop setup for metadata
setup_target = '''setupDragDrop('model-drop-zone', 'model-file', 'model-file-name');'''
new_setup = '''setupDragDrop('model-drop-zone', 'model-file', 'model-file-name');
            setupDragDrop('metadata-drop-zone', 'metadata-file', 'metadata-file-name');'''
html = html.replace(setup_target, new_setup)

with open('templates/dashboard.html', 'w', encoding='utf-8') as f:
    f.write(html)
print('Dashboard HTML fixed!')
