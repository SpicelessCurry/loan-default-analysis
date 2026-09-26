import re

with open('templates/dashboard.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Fix uploadModel response checking
model_target = '''const res = await fetch('/api/models/upload', { method: 'POST', body: formData });
                if (res.ok) {
                    showToast('Model uploaded successfully');'''

model_new = '''const res = await fetch('/api/models/upload', { method: 'POST', body: formData });
                const data = await res.json().catch(()=>({}));
                if (res.ok && data.success !== false) {
                    showToast('Model uploaded successfully');'''

html = html.replace(model_target, model_new)

# Fix error toast for model upload
model_err_target = '''} else {
                    const data = await res.json().catch(()=>({}));
                    showToast(data.error || 'Upload failed', 'error');
                }'''
model_err_new = '''} else {
                    showToast(data.message || data.error || 'Upload failed', 'error');
                }'''
html = html.replace(model_err_target, model_err_new)


# Fix uploadDataset response checking
dataset_target = '''const res = await fetch('/api/dataset/upload', { method: 'POST', body: formData });
                if (res.ok) {
                    showToast('Dataset uploaded and cached successfully');'''

dataset_new = '''const res = await fetch('/api/dataset/upload', { method: 'POST', body: formData });
                const data = await res.json().catch(()=>({}));
                if (res.ok && data.success !== false) {
                    showToast('Dataset uploaded and cached successfully');'''

html = html.replace(dataset_target, dataset_new)

# Fix error toast for dataset upload
dataset_err_target = '''} else {
                    const data = await res.json().catch(()=>({}));
                    showToast(data.error || 'Upload failed', 'error');
                }'''
dataset_err_new = '''} else {
                    showToast(data.message || data.error || 'Upload failed', 'error');
                }'''
html = html.replace(dataset_err_target, dataset_err_new)

with open('templates/dashboard.html', 'w', encoding='utf-8') as f:
    f.write(html)
print('Dashboard responses fixed!')
