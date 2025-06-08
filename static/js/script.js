// File upload handling and drag-drop functionality

document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.querySelector('.upload-area');
    const fileInput = document.getElementById('file-input');
    const fileList = document.getElementById('file-list');
    const uploadForm = document.getElementById('upload-form');
    const uploadBtn = document.getElementById('upload-btn');
    
    // Handle click on upload area
    if (uploadArea) {
        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });
        
        // Drag and drop functionality
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            handleFiles(files);
        });
    }
    
    // Handle file selection
    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            handleFiles(e.target.files);
        });
    }
    
    // Handle files
    function handleFiles(files) {
        fileList.innerHTML = '';
        let validFiles = [];
        
        for (let file of files) {
            if (file.type === 'application/pdf' || file.name.endsWith('.zip')) {
                validFiles.push(file);
                addFileToList(file);
            } else {
                showAlert('Only PDF and ZIP files are allowed', 'warning');
            }
        }
        
        if (validFiles.length > 0) {
            uploadBtn.disabled = false;
            
            // Update file input with valid files
            const dataTransfer = new DataTransfer();
            validFiles.forEach(file => dataTransfer.items.add(file));
            fileInput.files = dataTransfer.files;
        }
    }
    
    // Add file to display list
    function addFileToList(file) {
        const fileItem = document.createElement('div');
        fileItem.className = 'alert alert-info d-flex justify-content-between align-items-center';
        
        const fileInfo = document.createElement('span');
        fileInfo.textContent = `${file.name} (${formatFileSize(file.size)})`;
        
        const removeBtn = document.createElement('button');
        removeBtn.className = 'btn btn-sm btn-danger';
        removeBtn.textContent = 'Remove';
        removeBtn.onclick = () => {
            fileItem.remove();
            if (fileList.children.length === 0) {
                uploadBtn.disabled = true;
            }
        };
        
        fileItem.appendChild(fileInfo);
        fileItem.appendChild(removeBtn);
        fileList.appendChild(fileItem);
    }
    
    // Format file size
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    // Show alert message
    function showAlert(message, type = 'info') {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        const container = document.querySelector('.container');
        container.insertBefore(alertDiv, container.firstChild);
        
        setTimeout(() => {
            alertDiv.remove();
        }, 5000);
    }
    
    // Handle form submission
    if (uploadForm) {
        uploadForm.addEventListener('submit', (e) => {
            uploadBtn.disabled = true;
            uploadBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing...';
        });
    }
    
    // Copy similarity values on click
    const similarityCells = document.querySelectorAll('.similarity-cell');
    similarityCells.forEach(cell => {
        cell.style.cursor = 'pointer';
        cell.addEventListener('click', function() {
            const value = this.textContent;
            navigator.clipboard.writeText(value).then(() => {
                showAlert(`Copied: ${value}`, 'success');
            });
        });
    });
    
    // Add tooltips to AI score indicators
    const aiScores = document.querySelectorAll('.ai-score');
    aiScores.forEach(score => {
        const value = parseFloat(score.textContent);
        let message = '';
        
        if (value < 30) {
            message = 'Low probability of AI-generated content';
        } else if (value < 70) {
            message = 'Medium probability of AI-generated content';
        } else {
            message = 'High probability of AI-generated content';
        }
        
        score.setAttribute('title', message);
    });
    
    // Initialize Bootstrap tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});

// Export results as JSON
function exportResults() {
    const results = {
        files: [],
        similarity_matrix: [],
        average_ai_score: document.querySelector('.average-score').textContent
    };
    
    // Collect file results
    document.querySelectorAll('.file-result-card').forEach(card => {
        results.files.push({
            name: card.querySelector('.card-title').textContent,
            ai_score: card.querySelector('.ai-score').textContent
        });
    });
    
    // Collect similarity matrix
    const matrix = [];
    document.querySelectorAll('.similarity-matrix tbody tr').forEach(row => {
        const rowData = [];
        row.querySelectorAll('td').forEach((cell, index) => {
            if (index > 0) { // Skip first cell (file name)
                rowData.push(cell.textContent);
            }
        });
        matrix.push(rowData);
    });
    results.similarity_matrix = matrix;
    
    // Download as JSON
    const dataStr = JSON.stringify(results, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = 'authenticity_results.json';
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
}