document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const uploadArea = document.getElementById('upload-area');
    const uploadContent = document.getElementById('upload-content');
    const fileInput = document.getElementById('file-input');
    const previewImage = document.getElementById('preview-image');
    const analyzeButton = document.getElementById('analyze-button');
    const resultsContainer = document.getElementById('results-container');
    const diagnosisResult = document.getElementById('diagnosis-result');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceValue = document.getElementById('confidence-value');
    const detailedResults = document.getElementById('detailed-results');
    const newAnalysisButton = document.getElementById('new-analysis');

    // Variables
    let selectedFile = null;

    // Event Listeners
    uploadArea.addEventListener('click', () => fileInput.click());
    
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
        
        if (e.dataTransfer.files.length) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFileSelect(e.target.files[0]);
        }
    });
    
    analyzeButton.addEventListener('click', analyzeImage);
    
    newAnalysisButton.addEventListener('click', resetUI);

    // Functions
    function handleFileSelect(file) {
        if (!file.type.match('image.*')) {
            alert('Please select an image file');
            return;
        }
        
        selectedFile = file;
        const reader = new FileReader();
        
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadContent.classList.add('hidden');
            previewImage.classList.remove('hidden');
            analyzeButton.disabled = false;
        };
        
        reader.readAsDataURL(file);
    }
    
    async function analyzeImage() {
        if (!selectedFile) return;
        
        analyzeButton.disabled = true;
        analyzeButton.textContent = 'Analyzing...';
        
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            
            const result = await response.json();
            displayResults(result);
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while analyzing the image. Please try again.');
            analyzeButton.disabled = false;
            analyzeButton.textContent = 'Analyze Image';
        }
    }
    
    function displayResults(result) {
        // Display diagnosis
        diagnosisResult.textContent = formatLabel(result.label);
        
        // Display confidence
        const confidencePercent = Math.round(result.confidence * 100);
        confidenceBar.style.width = `${confidencePercent}%`;
        confidenceValue.textContent = `${confidencePercent}%`;
        
        // Display detailed results
        detailedResults.innerHTML = '';
        
        Object.entries(result.probs).forEach(([label, probability]) => {
            const probabilityPercent = Math.round(probability * 100);
            
            const item = document.createElement('div');
            item.className = 'probability-item';
            
            const labelElement = document.createElement('span');
            labelElement.textContent = formatLabel(label);
            
            const barContainer = document.createElement('div');
            barContainer.className = 'probability-bar-container';
            
            const bar = document.createElement('div');
            bar.className = 'probability-bar';
            bar.style.width = `${probabilityPercent}%`;
            
            const valueElement = document.createElement('span');
            valueElement.textContent = `${probabilityPercent}%`;
            
            barContainer.appendChild(bar);
            item.appendChild(labelElement);
            item.appendChild(barContainer);
            item.appendChild(valueElement);
            
            detailedResults.appendChild(item);
        });
        
        // Show results
        resultsContainer.classList.remove('hidden');
        analyzeButton.textContent = 'Analyze Image';
    }
    
    function formatLabel(label) {
        return label.replace(/_/g, ' ').replace(/___/g, ' - ');
    }
    
    function resetUI() {
        selectedFile = null;
        previewImage.src = '';
        previewImage.classList.add('hidden');
        uploadContent.classList.remove('hidden');
        resultsContainer.classList.add('hidden');
        analyzeButton.disabled = true;
        analyzeButton.textContent = 'Analyze Image';
    }
});