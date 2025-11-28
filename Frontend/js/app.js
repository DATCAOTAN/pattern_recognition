/**
 * EcoSort AI - Main JavaScript Application
 * Waste Classification System with YOLO
 */

// ============== Configuration ==============
const API_BASE_URL = 'http://localhost:8000';
let WS_URL = 'ws://localhost:8000/ws/stream';

// ============== Global State ==============
let state = {
    isRunning: false,
    currentSource: 'upload',
    confidenceThreshold: 0.75,
    classFilter: ['bottle', 'can', 'bag', 'banana_peel', 'eggshell', 'leaves'],
    detections: [],
    logs: [],
    statistics: {
        total: 0,
        inorganic: 0,
        organic: 0,
        classCounts: {}
    },
    stream: null,
    websocket: null,
    animationFrame: null
};

// ============== Chart Instances ==============
let statsChart = null;
let pieChart = null;

// ============== Initialization ==============
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🌿 EcoSort AI Frontend Initialized');
    
    // Initialize charts
    initCharts();
    
    // Check system status
    await checkSystemStatus();
    
    // Start periodic status updates
    setInterval(checkSystemStatus, 5000);
    
    // Setup drag and drop
    setupDragDrop();
});

// ============== Chart Initialization ==============
function initCharts() {
    // Stats Bar Chart
    const statsCtx = document.getElementById('statsChart').getContext('2d');
    statsChart = new Chart(statsCtx, {
        type: 'bar',
        data: {
            labels: ['Inorganic', 'Organic'],
            datasets: [{
                label: 'Items',
                data: [0, 0],
                backgroundColor: ['#00ff00', '#ff6600'],
                borderColor: ['#00cc00', '#cc5500'],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: { color: '#b8b8b8' },
                    grid: { color: '#2a4a6a' }
                },
                x: {
                    ticks: { color: '#b8b8b8' },
                    grid: { display: false }
                }
            }
        }
    });

    // Pie Chart
    const pieCtx = document.getElementById('pieChart').getContext('2d');
    pieChart = new Chart(pieCtx, {
        type: 'doughnut',
        data: {
            labels: ['Inorganic', 'Organic', 'Material', '6 classes'],
            datasets: [{
                data: [35, 38, 18, 12],
                backgroundColor: ['#00ff00', '#ff6600', '#00d4ff', '#888888'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            cutout: '60%'
        }
    });
}

// ============== API Functions ==============
async function checkSystemStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/system/status`);
        const data = await response.json();
        
        updateSystemStatusUI(data);
    } catch (error) {
        console.error('Error checking system status:', error);
        updateSystemStatusUI({
            health: 'Error',
            model_status: 'Disconnected',
            model_path: null,
            gpu: { available: false }
        });
    }
}

function updateSystemStatusUI(data) {
    // Model status
    const modelStatusEl = document.getElementById('modelStatus');
    const statusIndicator = modelStatusEl.querySelector('.status-indicator');
    
    if (data.model_status === 'Ready') {
        statusIndicator.className = 'status-indicator ready';
        modelStatusEl.innerHTML = `<span class="status-indicator ready"></span>Ready (best.pt loaded)`;
    } else if (data.model_status === 'Failed to load') {
        statusIndicator.className = 'status-indicator error';
        modelStatusEl.innerHTML = `<span class="status-indicator error"></span>Failed to load`;
    } else {
        statusIndicator.className = 'status-indicator loading';
        modelStatusEl.innerHTML = `<span class="status-indicator loading"></span>${data.model_status}`;
    }
    
    // System health
    const healthDot = document.getElementById('healthDot');
    const systemHealth = document.getElementById('systemHealth');
    systemHealth.textContent = data.health;
    healthDot.className = `status-dot ${data.health.toLowerCase()}`;
    
    // Model path
    document.getElementById('modelPath').textContent = data.model_path || '/models/best.pt';
    
    // GPU usage
    const gpuUsage = document.getElementById('gpuUsage');
    if (data.gpu && data.gpu.available) {
        gpuUsage.textContent = data.gpu.gpu_load_percent || '0';
    } else {
        gpuUsage.textContent = 'N/A';
    }
}

async function predictImage(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch(
            `${API_BASE_URL}/predict/image?confidence=${state.confidenceThreshold}&draw_boxes=true`,
            {
                method: 'POST',
                body: formData
            }
        );
        
        const data = await response.json();
        
        if (data.success) {
            displayDetectionResult(data);
            updateStatistics(data.detections);
            addLogsFromDetections(data.detections);
            updateSortingDecision(data.sorting_decision);
        }
        
        return data;
    } catch (error) {
        console.error('Error predicting image:', error);
        alert('Error connecting to server. Make sure backend is running.');
    }
}

async function predictFrame(frameData) {
    try {
        const response = await fetch(
            `${API_BASE_URL}/predict/frame?confidence=${state.confidenceThreshold}`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ frame_data: frameData })
            }
        );
        
        return await response.json();
    } catch (error) {
        console.error('Error predicting frame:', error);
        return null;
    }
}

// ============== Source Selection ==============
function changeSource() {
    const source = document.getElementById('sourceSelect').value;
    state.currentSource = source;
    
    const cameraPreview = document.getElementById('cameraPreview');
    const uploadArea = document.getElementById('uploadArea');
    const videoElement = document.getElementById('videoElement');
    const placeholder = document.getElementById('cameraPlaceholder');
    
    // Stop any running stream
    stopCamera();
    
    switch(source) {
        case 'camera':
            cameraPreview.style.display = 'flex';
            uploadArea.style.display = 'none';
            placeholder.style.display = 'flex';
            break;
        case 'video':
        case 'image':
        case 'upload':
            cameraPreview.style.display = 'none';
            uploadArea.style.display = 'block';
            break;
    }
}

// ============== File Upload ==============
function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const isImage = file.type.startsWith('image/');
    const isVideo = file.type.startsWith('video/');
    
    if (isImage) {
        processImageFile(file);
    } else if (isVideo) {
        processVideoFile(file);
    } else {
        alert('Unsupported file type. Please upload an image or video.');
    }
}

function processImageFile(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const img = document.getElementById('detectionImage');
        img.src = e.target.result;
        img.classList.add('active');
        document.getElementById('placeholderView').classList.add('hidden');
    };
    reader.readAsDataURL(file);
    
    // Send to API
    predictImage(file);
}

function processVideoFile(file) {
    alert('Video processing started. This may take a moment...');
    // For videos, we'll process frame by frame
    // This is a simplified version - full implementation would stream frames
}

function setupDragDrop() {
    const uploadArea = document.getElementById('uploadArea');
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#00d4ff';
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = '';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '';
        
        const file = e.dataTransfer.files[0];
        if (file) {
            handleFileUpload({ target: { files: [file] } });
        }
    });
}

// ============== Camera Functions ==============
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 } 
        });
        
        const videoElement = document.getElementById('videoElement');
        videoElement.srcObject = stream;
        videoElement.classList.add('active');
        document.getElementById('cameraPlaceholder').style.display = 'none';
        
        state.stream = stream;
        return true;
    } catch (error) {
        console.error('Error accessing camera:', error);
        alert('Could not access camera. Please check permissions.');
        return false;
    }
}

function stopCamera() {
    if (state.stream) {
        state.stream.getTracks().forEach(track => track.stop());
        state.stream = null;
    }
    
    const videoElement = document.getElementById('videoElement');
    videoElement.srcObject = null;
    videoElement.classList.remove('active');
    document.getElementById('cameraPlaceholder').style.display = 'flex';
}

function captureFrame() {
    const video = document.getElementById('videoElement');
    const canvas = document.getElementById('cameraCanvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    
    return canvas.toDataURL('image/jpeg', 0.8);
}

// ============== Classification Control ==============
async function startClassification() {
    state.isRunning = true;
    
    document.getElementById('startBtn').style.display = 'none';
    document.getElementById('stopBtn').style.display = 'flex';
    
    if (state.currentSource === 'camera') {
        const started = await startCamera();
        if (started) {
            startRealTimeProcessing();
        }
    }
}

function stopClassification() {
    state.isRunning = false;
    
    document.getElementById('startBtn').style.display = 'flex';
    document.getElementById('stopBtn').style.display = 'none';
    
    if (state.animationFrame) {
        cancelAnimationFrame(state.animationFrame);
        state.animationFrame = null;
    }
    
    stopCamera();
    
    if (state.websocket) {
        state.websocket.close();
        state.websocket = null;
    }
}

function startRealTimeProcessing() {
    const processFrame = async () => {
        if (!state.isRunning) return;
        
        const frameData = captureFrame();
        const result = await predictFrame(frameData);
        
        if (result && result.success) {
            displayDetectionResult(result);
            updateStatistics(result.detections);
            if (result.detections.length > 0) {
                addLogsFromDetections(result.detections);
            }
            updateSortingDecision(result.sorting_decision);
        }
        
        // Continue processing at ~10 FPS
        setTimeout(() => {
            state.animationFrame = requestAnimationFrame(processFrame);
        }, 100);
    };
    
    processFrame();
}

// ============== Display Functions ==============
function displayDetectionResult(data) {
    const img = document.getElementById('detectionImage');
    const placeholder = document.getElementById('placeholderView');
    
    if (data.processed_image) {
        img.src = data.processed_image;
        img.classList.add('active');
        placeholder.classList.add('hidden');
    }
}

function updateSortingDecision(decision) {
    const topDecision = document.getElementById('sortingDecisionTop');
    const bottomDecision = document.getElementById('sortingDecisionBottom');
    const signalGreen = document.getElementById('signalGreen');
    const signalRed = document.getElementById('signalRed');
    
    let decisionText = 'WAITING';
    
    switch(decision.signal) {
        case 'GREEN':
            decisionText = 'INORGANIC STREAM';
            signalGreen.classList.add('active');
            signalRed.classList.remove('active');
            break;
        case 'RED':
            decisionText = 'ORGANIC STREAM';
            signalGreen.classList.remove('active');
            signalRed.classList.add('active');
            break;
        case 'MIXED':
            decisionText = 'SEPARATE STREAMS';
            signalGreen.classList.add('active');
            signalRed.classList.add('active');
            break;
        default:
            decisionText = 'WAITING';
            signalGreen.classList.remove('active');
            signalRed.classList.remove('active');
    }
    
    topDecision.querySelector('.decision-text').textContent = `Sorting Decision: ${decisionText}`;
    bottomDecision.querySelector('.decision-text').textContent = `Sorting Decision: ${decisionText}`;
}

// ============== Statistics ==============
function updateStatistics(detections) {
    // Count by category
    const inorganic = detections.filter(d => d.category === 'Inorganic').length;
    const organic = detections.filter(d => d.category === 'Organic').length;
    
    state.statistics.total += detections.length;
    state.statistics.inorganic += inorganic;
    state.statistics.organic += organic;
    
    // Count by class
    detections.forEach(d => {
        const cls = d.class_name;
        state.statistics.classCounts[cls] = (state.statistics.classCounts[cls] || 0) + 1;
    });
    
    // Update UI
    document.getElementById('totalItems').textContent = state.statistics.total;
    document.getElementById('inorganicCount').textContent = state.statistics.inorganic;
    document.getElementById('organicCount').textContent = state.statistics.organic;
    
    // Update charts
    updateCharts();
}

function updateCharts() {
    // Update bar chart
    statsChart.data.datasets[0].data = [
        state.statistics.inorganic,
        state.statistics.organic
    ];
    statsChart.update('none');
    
    // Update pie chart
    const total = state.statistics.total || 1;
    pieChart.data.datasets[0].data = [
        Math.round(state.statistics.inorganic / total * 100),
        Math.round(state.statistics.organic / total * 100),
        18,
        12
    ];
    pieChart.update('none');
}

// ============== Logging ==============
function addLogsFromDetections(detections) {
    const logTable = document.getElementById('logTable');
    const placeholder = logTable.querySelector('.log-placeholder');
    
    if (placeholder) {
        placeholder.remove();
    }
    
    detections.forEach(det => {
        const now = new Date();
        const timeStr = now.toISOString().replace('T', ' ').substring(0, 19);
        
        const logEntry = document.createElement('div');
        logEntry.className = 'log-entry';
        logEntry.innerHTML = `
            <span class="log-time">${timeStr}</span>
            <span class="log-class ${det.category.toLowerCase()}">${det.class_name.replace('_', ' ')} (${det.category})</span>
        `;
        
        logTable.insertBefore(logEntry, logTable.firstChild);
        
        // Keep only last 100 entries
        while (logTable.children.length > 100) {
            logTable.removeChild(logTable.lastChild);
        }
        
        state.logs.push({
            time: timeStr,
            class_name: det.class_name,
            category: det.category,
            confidence: det.confidence
        });
    });
}

function filterLogs() {
    const searchTerm = document.getElementById('logSearch').value.toLowerCase();
    const entries = document.querySelectorAll('.log-entry');
    
    entries.forEach(entry => {
        const text = entry.textContent.toLowerCase();
        entry.style.display = text.includes(searchTerm) ? 'flex' : 'none';
    });
}

// ============== Configuration ==============
function updateConfidence(value) {
    state.confidenceThreshold = value / 100;
    document.getElementById('confidenceValue').textContent = (value / 100).toFixed(2);
}

function updateClassFilter() {
    const checkboxes = document.querySelectorAll('.class-filters input[type="checkbox"]');
    state.classFilter = [];
    
    checkboxes.forEach(cb => {
        if (cb.checked) {
            state.classFilter.push(cb.dataset.class);
        }
    });
}

// ============== Snapshot ==============
async function takeSnapshot() {
    const img = document.getElementById('detectionImage');
    
    if (!img.src || img.src === '') {
        alert('No image to capture');
        return;
    }
    
    try {
        const response = await fetch(
            `${API_BASE_URL}/snapshot?include_detections=true`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ frame_data: img.src })
            }
        );
        
        const data = await response.json();
        
        if (data.success) {
            alert(`Snapshot saved: ${data.filename}`);
        }
    } catch (error) {
        // Fallback: download directly
        const link = document.createElement('a');
        link.download = `snapshot_${Date.now()}.jpg`;
        link.href = img.src;
        link.click();
    }
}

// ============== Export Functions ==============
function exportLogs() {
    document.getElementById('exportModal').classList.add('active');
}

function closeModal() {
    document.getElementById('exportModal').classList.remove('active');
}

async function downloadCSV() {
    try {
        window.open(`${API_BASE_URL}/logs/export/csv`, '_blank');
    } catch (error) {
        // Fallback: generate CSV locally
        let csv = 'Time,Class,Category,Confidence\n';
        state.logs.forEach(log => {
            csv += `${log.time},${log.class_name},${log.category},${log.confidence}\n`;
        });
        
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'detection_logs.csv';
        link.click();
    }
    closeModal();
}

async function downloadExcel() {
    try {
        window.open(`${API_BASE_URL}/logs/export/excel`, '_blank');
    } catch (error) {
        alert('Could not export to Excel. Please try CSV export.');
    }
    closeModal();
}

// ============== Utility Functions ==============
function logout() {
    if (confirm('Are you sure you want to logout?')) {
        stopClassification();
        location.reload();
    }
}

// ============== Page Navigation ==============
document.querySelectorAll('.nav-links a[data-page]').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        document.querySelectorAll('.nav-links a').forEach(l => l.classList.remove('active'));
        link.classList.add('active');
        
        // Page switching logic would go here
        console.log(`Navigating to: ${link.dataset.page}`);
    });
});
