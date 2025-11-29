/**
 * EcoSort AI - Main JavaScript Application
 * Waste Classification System with YOLO
 * Full features: Image, Video, Camera processing
 */

// ============== Configuration ==============
const API_BASE_URL = 'http://localhost:8000';

// ============== Global State ==============
let isProcessing = false;
let currentSource = 'upload';
let confidenceThreshold = 0.75;
let classFilters = {
    bottle: true,
    can: true,
    bag: true,
    banana_peel: true,
    eggshell: true,
    leaves: true
};
let detectionLogs = [];
let statsChart = null;
let pieChart = null;
let cameraStream = null;
let animationFrameId = null;
let uploadedVideo = null;

// Statistics
let stats = {
    total: 0,
    inorganic: 0,
    organic: 0,
    byClass: {
        bottle: 0,
        can: 0,
        bag: 0,
        banana_peel: 0,
        eggshell: 0,
        leaves: 0
    }
};

// ==================== INITIALIZATION ====================

document.addEventListener('DOMContentLoaded', function() {
    console.log('🌿 EcoSort AI Frontend Initialized');
    initializeCharts();
    checkSystemStatus();
    setInterval(checkSystemStatus, 5000);
    setupDragAndDrop();
});

function initializeCharts() {
    // Bar Chart for Session Stats
    const statsCtx = document.getElementById('statsChart').getContext('2d');
    statsChart = new Chart(statsCtx, {
        type: 'bar',
        data: {
            labels: ['Inorganic', 'Organic'],
            datasets: [{
                data: [0, 0],
                backgroundColor: ['#00ff00', '#ff6600'],
                borderRadius: 5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: '#333' },
                    ticks: { color: '#b8b8b8' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#b8b8b8' }
                }
            }
        }
    });

    // Pie Chart for Material Breakdown
    const pieCtx = document.getElementById('pieChart').getContext('2d');
    pieChart = new Chart(pieCtx, {
        type: 'doughnut',
        data: {
            labels: ['Bottle', 'Can', 'Bag', 'Banana Peel', 'Eggshell', 'Leaves'],
            datasets: [{
                data: [0, 0, 0, 0, 0, 0],
                backgroundColor: ['#22c55e', '#3b82f6', '#06b6d4', '#f97316', '#eab308', '#84cc16']
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            cutout: '60%'
        }
    });
}

// ==================== SYSTEM STATUS ====================

async function checkSystemStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/system/status`);
        const data = await response.json();
        
        // Update Model Status
        const modelStatus = document.getElementById('modelStatus');
        if (data.model_status === 'Ready') {
            modelStatus.innerHTML = `<span class="status-indicator ready"></span>Ready (best.pt loaded)`;
        } else {
            modelStatus.innerHTML = `<span class="status-indicator error"></span>${data.model_status}`;
        }
        
        // Update Footer
        document.getElementById('systemHealth').textContent = data.health || 'Optimal';
        document.getElementById('healthDot').className = `status-dot ${(data.health || 'optimal').toLowerCase()}`;
        document.getElementById('gpuUsage').textContent = data.gpu?.gpu_load_percent || '--';
        document.getElementById('modelPath').textContent = data.model_path || '/models/best.pt';
        
    } catch (error) {
        console.error('System status error:', error);
        document.getElementById('modelStatus').innerHTML = 
            `<span class="status-indicator error"></span>Server Disconnected`;
        document.getElementById('systemHealth').textContent = 'Offline';
        document.getElementById('healthDot').className = 'status-dot critical';
    }
}

// ==================== SOURCE HANDLING ====================

function changeSource() {
    const source = document.getElementById('sourceSelect').value;
    currentSource = source;
    
    // Stop any ongoing processing
    stopClassification();
    stopCamera();
    
    // Reset display completely
    resetDisplay();
    
    // Hide all input areas first
    document.getElementById('uploadArea').style.display = 'none';
    document.getElementById('cameraPreview').style.display = 'none';
    
    // Show relevant input area
    switch(source) {
        case 'image':
            document.getElementById('uploadArea').style.display = 'block';
            document.getElementById('fileInput').accept = 'image/*';
            break;
        case 'video':
            document.getElementById('uploadArea').style.display = 'block';
            document.getElementById('fileInput').accept = 'video/*';
            break;
        case 'camera':
            document.getElementById('cameraPreview').style.display = 'block';
            initializeCamera();
            break;
        default: // upload
            document.getElementById('uploadArea').style.display = 'block';
            document.getElementById('fileInput').accept = 'image/*,video/*';
    }
}

/**
 * Reset all display elements to initial state
 */
function resetDisplay() {
    // Hide and reset video display
    hideVideoDisplay();
    
    // Reset image
    const img = document.getElementById('detectionImage');
    if (img) {
        img.src = '';
        img.style.display = 'none';
    }
    
    // Reset canvas
    const canvas = document.getElementById('detectionCanvas');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        canvas.style.display = 'none';
    }
    
    // Reset stored data
    uploadedImageFile = null;
    lastImageDetections = [];
    originalImageSrc = null;
    
    // Show placeholder
    const placeholder = document.getElementById('placeholderView');
    if (placeholder) {
        placeholder.style.display = 'flex';
    }
    
    // Reset file input
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.value = '';
    }
    
    console.log('Display reset complete');
}

// ==================== CAMERA HANDLING ====================

async function initializeCamera() {
    const placeholder = document.getElementById('cameraPlaceholder');
    const video = document.getElementById('videoElement');
    
    try {
        placeholder.innerHTML = '<i class="fas fa-spinner fa-spin"></i><span>Connecting to camera...</span>';
        placeholder.style.display = 'flex';
        video.style.display = 'none';
        
        // Request camera access
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'environment'
            },
            audio: false
        });
        
        video.srcObject = cameraStream;
        await video.play();
        
        video.style.display = 'block';
        placeholder.style.display = 'none';
        
        console.log('✅ Camera initialized successfully');
        
    } catch (error) {
        console.error('❌ Camera error:', error);
        placeholder.innerHTML = 
            `<i class="fas fa-exclamation-triangle"></i><span>Camera Error: ${error.message}</span>`;
        placeholder.style.display = 'flex';
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    
    const video = document.getElementById('videoElement');
    if (video) {
        video.srcObject = null;
        video.style.display = 'none';
    }
    
    const placeholder = document.getElementById('cameraPlaceholder');
    if (placeholder) {
        placeholder.innerHTML = '<i class="fas fa-video"></i><span>Camera Preview</span>';
        placeholder.style.display = 'flex';
    }
}

// ==================== FILE UPLOAD HANDLING ====================

function setupDragAndDrop() {
    const uploadArea = document.getElementById('uploadArea');
    
    if (!uploadArea) return;
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#00d4ff';
        uploadArea.style.background = 'rgba(0, 212, 255, 0.1)';
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = '';
        uploadArea.style.background = '';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '';
        uploadArea.style.background = '';
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
}

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    const isImage = file.type.startsWith('image/');
    const isVideo = file.type.startsWith('video/');
    
    if (isImage) {
        currentSource = 'image';
        displayImage(file);
    } else if (isVideo) {
        currentSource = 'video';
        displayVideo(file);
    } else {
        alert('Unsupported file type. Please upload an image or video.');
    }
}

// Store the uploaded file for later use
let uploadedImageFile = null;
let lastImageDetections = []; // Store detections for filter re-drawing
let originalImageSrc = null; // Store original image for re-drawing

function displayImage(file) {
    hideVideoDisplay();
    uploadedImageFile = file;
    lastImageDetections = []; // Reset detections
    
    const reader = new FileReader();
    reader.onload = function(e) {
        const img = document.getElementById('detectionImage');
        const canvas = document.getElementById('detectionCanvas');
        
        originalImageSrc = e.target.result; // Store original
        img.src = e.target.result;
        img.style.display = 'block';
        img.style.maxWidth = '100%';
        img.style.maxHeight = '100%';
        img.style.objectFit = 'contain';
        
        // Hide canvas until processing
        canvas.style.display = 'none';
        
        img.onload = function() {
            console.log(`Image loaded: ${img.naturalWidth}x${img.naturalHeight}`);
        };
        document.getElementById('placeholderView').style.display = 'none';
    };
    reader.readAsDataURL(file);
}

/**
 * Draw image with filtered bounding boxes on canvas
 */
function drawImageWithDetections(img, detections) {
    const canvas = document.getElementById('detectionCanvas');
    const ctx = canvas.getContext('2d');
    const displayArea = document.querySelector('.detection-display');
    
    // Set canvas size to match image
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    
    // Position canvas to overlay on image area
    canvas.style.position = 'absolute';
    canvas.style.top = '50%';
    canvas.style.left = '50%';
    canvas.style.transform = 'translate(-50%, -50%)';
    canvas.style.maxWidth = '100%';
    canvas.style.maxHeight = '100%';
    canvas.style.width = 'auto';
    canvas.style.height = 'auto';
    canvas.style.display = 'block';
    canvas.style.zIndex = '10';
    
    // Draw the original image on canvas
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    
    // Filter and draw bounding boxes
    const filteredDetections = detections.filter(d => classFilters[d.class_name]);
    filteredDetections.forEach(det => drawBoundingBox(ctx, det));
    
    // Hide the img element since we're using canvas
    img.style.display = 'none';
    
    console.log(`Drew ${filteredDetections.length}/${detections.length} detections (filtered by class)`);
}

function displayVideo(file) {
    const img = document.getElementById('detectionImage');
    img.style.display = 'none';
    document.getElementById('placeholderView').style.display = 'none';
    
    // Create or get video display element
    const displayArea = document.querySelector('.detection-display');
    let videoDisplay = document.getElementById('uploadedVideoDisplay');
    
    if (!videoDisplay) {
        videoDisplay = document.createElement('video');
        videoDisplay.id = 'uploadedVideoDisplay';
        videoDisplay.style.cssText = 'width: 100%; height: 100%; object-fit: contain; background: #000;';
        videoDisplay.muted = true;
        videoDisplay.playsInline = true;
        displayArea.insertBefore(videoDisplay, displayArea.firstChild);
    }
    
    videoDisplay.src = URL.createObjectURL(file);
    videoDisplay.style.display = 'block';
    
    videoDisplay.onloadedmetadata = function() {
        console.log(`Video loaded: ${videoDisplay.videoWidth}x${videoDisplay.videoHeight}`);
        resizeCanvas();
    };
    
    uploadedVideo = videoDisplay;
}

function hideVideoDisplay() {
    const videoDisplay = document.getElementById('uploadedVideoDisplay');
    if (videoDisplay) {
        videoDisplay.pause();
        videoDisplay.src = '';
        videoDisplay.style.display = 'none';
    }
    uploadedVideo = null;
}

// ==================== CLASSIFICATION CONTROL ====================

function startClassification() {
    if (isProcessing) return;
    
    isProcessing = true;
    document.getElementById('startBtn').style.display = 'none';
    document.getElementById('stopBtn').style.display = 'flex';
    
    updateSortingDecision('PROCESSING...');
    
    console.log(`Starting classification for source: ${currentSource}`);
    
    switch(currentSource) {
        case 'image':
        case 'upload':
            processImage();
            break;
        case 'video':
            processVideo();
            break;
        case 'camera':
            processCameraStream();
            break;
    }
}

function stopClassification() {
    isProcessing = false;
    document.getElementById('startBtn').style.display = 'flex';
    document.getElementById('stopBtn').style.display = 'none';
    
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    
    // Pause uploaded video
    const videoDisplay = document.getElementById('uploadedVideoDisplay');
    if (videoDisplay) {
        videoDisplay.pause();
    }
    
    // Clear canvas
    const canvas = document.getElementById('detectionCanvas');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    
    updateSortingDecision('STOPPED');
    console.log('Classification stopped');
}

// ==================== IMAGE PROCESSING ====================

async function processImage() {
    const img = document.getElementById('detectionImage');
    const canvas = document.getElementById('detectionCanvas');
    const ctx = canvas.getContext('2d');
    
    // Check if we have an uploaded file or image source
    if (!uploadedImageFile && (!img.src || img.src === '' || img.src === window.location.href)) {
        alert('Please upload an image first');
        stopClassification();
        return;
    }
    
    try {
        // Wait for image to load if needed
        if (!img.complete) {
            await new Promise(resolve => {
                img.onload = resolve;
                setTimeout(resolve, 1000); // Timeout fallback
            });
        }
        
        console.log(`Processing image: ${img.naturalWidth}x${img.naturalHeight}`);
        
        // Prepare FormData - use the original file if available
        const formData = new FormData();
        if (uploadedImageFile) {
            formData.append('file', uploadedImageFile, uploadedImageFile.name);
        } else {
            // Fallback: convert image src to blob
            const response = await fetch(img.src);
            const blob = await response.blob();
            formData.append('file', blob, 'image.jpg');
        }
        
        // Send to API - DON'T draw boxes on backend, we'll draw filtered boxes on frontend
        console.log(`Sending to API: ${API_BASE_URL}/predict/image?confidence=${confidenceThreshold}&draw_boxes=false`);
        
        const result = await fetch(`${API_BASE_URL}/predict/image?confidence=${confidenceThreshold}&draw_boxes=false`, {
            method: 'POST',
            body: formData
        });
        
        if (!result.ok) {
            const errorText = await result.text();
            console.error('API Error:', result.status, errorText);
            throw new Error(`API returned ${result.status}: ${errorText}`);
        }
        
        const data = await result.json();
        console.log('API Response:', data);
        console.log('Detections count:', data.detections?.length || 0);
        
        if (data.success) {
            // Store detections for re-drawing when filter changes
            lastImageDetections = data.detections || [];
            
            // Draw image and filtered bounding boxes on canvas
            drawImageWithDetections(img, lastImageDetections);
            
            // Update stats and logs
            if (data.detections && data.detections.length > 0) {
                updateStats(data.detections);
                addToLog(data.detections);
                console.log(`✅ Detected ${data.detections.length} objects in image`);
            } else {
                console.log('No objects detected in image');
            }
            
            handleSortingDecision(data.sorting_decision);
        } else {
            console.error('Detection failed:', data);
            alert('Detection failed: ' + (data.detail || 'Unknown error'));
        }
        
    } catch (error) {
        console.error('Image processing error:', error);
        alert('Error processing image. Check if backend server is running on ' + API_BASE_URL);
    }
    
    // Don't auto-stop for image - let user see results
    isProcessing = false;
    document.getElementById('startBtn').style.display = 'flex';
    document.getElementById('stopBtn').style.display = 'none';
    updateSortingDecision(document.querySelector('.decision-text')?.textContent?.replace('Sorting Decision: ', '') || 'COMPLETED');
}

// ==================== VIDEO PROCESSING ====================

async function processVideo() {
    const videoDisplay = document.getElementById('uploadedVideoDisplay');
    if (!videoDisplay || !videoDisplay.src) {
        alert('Please upload a video first');
        stopClassification();
        return;
    }
    
    const canvas = document.getElementById('detectionCanvas');
    const ctx = canvas.getContext('2d');
    
    // Hide the video element, we'll draw everything on canvas
    videoDisplay.style.opacity = '0';
    videoDisplay.style.position = 'absolute';
    
    // Position canvas to fill detection display
    canvas.style.position = 'absolute';
    canvas.style.top = '50%';
    canvas.style.left = '50%';
    canvas.style.transform = 'translate(-50%, -50%)';
    canvas.style.maxWidth = '100%';
    canvas.style.maxHeight = '100%';
    canvas.style.display = 'block';
    canvas.style.zIndex = '10';
    
    // Start video playback
    videoDisplay.currentTime = 0;
    await videoDisplay.play();
    
    // Set canvas size to match video
    canvas.width = videoDisplay.videoWidth || 640;
    canvas.height = videoDisplay.videoHeight || 480;
    
    let frameCount = 0;
    let lastDetections = [];
    
    async function processFrame() {
        if (!isProcessing || videoDisplay.paused || videoDisplay.ended) {
            if (videoDisplay.ended) {
                updateSortingDecision('VIDEO COMPLETED');
            }
            videoDisplay.style.opacity = '1';
            videoDisplay.style.position = '';
            stopClassification();
            return;
        }
        
        frameCount++;
        
        // Always draw the current video frame on canvas
        ctx.drawImage(videoDisplay, 0, 0, canvas.width, canvas.height);
        
        // Draw last known detections
        if (lastDetections.length > 0) {
            const filteredDetections = lastDetections.filter(d => classFilters[d.class_name]);
            filteredDetections.forEach(det => drawBoundingBox(ctx, det));
        }
        
        // Process every 5th frame for API call (performance)
        if (frameCount % 5 === 0) {
            try {
                // Convert frame to blob
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = canvas.width;
                tempCanvas.height = canvas.height;
                const tempCtx = tempCanvas.getContext('2d');
                tempCtx.drawImage(videoDisplay, 0, 0, tempCanvas.width, tempCanvas.height);
                
                const blob = await new Promise(resolve => tempCanvas.toBlob(resolve, 'image/jpeg', 0.7));
                
                // Send to API
                const formData = new FormData();
                formData.append('file', blob, 'frame.jpg');
                
                const result = await fetch(`${API_BASE_URL}/predict/image?confidence=${confidenceThreshold}&draw_boxes=false`, {
                    method: 'POST',
                    body: formData
                });
                
                const data = await result.json();
                
                if (data.success) {
                    lastDetections = data.detections || [];
                    if (lastDetections.length > 0) {
                        updateStats(data.detections);
                        addToLog(data.detections);
                    }
                    handleSortingDecision(data.sorting_decision);
                }
                
            } catch (error) {
                console.error('Frame processing error:', error);
            }
        }
        
        // Continue processing
        animationFrameId = requestAnimationFrame(processFrame);
    }
    
    processFrame();
}

// ==================== CAMERA STREAM PROCESSING ====================

async function processCameraStream() {
    const video = document.getElementById('videoElement');
    const canvas = document.getElementById('detectionCanvas');
    const ctx = canvas.getContext('2d');
    
    if (!cameraStream || !video.srcObject) {
        alert('Camera not initialized. Please select Camera source first.');
        stopClassification();
        return;
    }
    
    // Position canvas in center of detection display
    canvas.style.position = 'absolute';
    canvas.style.top = '50%';
    canvas.style.left = '50%';
    canvas.style.transform = 'translate(-50%, -50%)';
    canvas.style.maxWidth = '100%';
    canvas.style.maxHeight = '100%';
    canvas.style.display = 'block';
    canvas.style.zIndex = '10';
    
    // Set canvas size to match video
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    
    // Hide placeholder and image
    document.getElementById('placeholderView').style.display = 'none';
    document.getElementById('detectionImage').style.display = 'none';
    
    let frameCount = 0;
    let lastDetections = [];
    
    async function processFrame() {
        if (!isProcessing || !cameraStream) {
            canvas.style.display = 'none';
            stopClassification();
            return;
        }
        
        frameCount++;
        
        // Always draw the current camera frame on canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Always draw last known detections on top
        if (lastDetections.length > 0) {
            const filteredDetections = lastDetections.filter(d => classFilters[d.class_name]);
            filteredDetections.forEach(det => drawBoundingBox(ctx, det));
        }
        
        // Process every 5th frame for API call (~6 FPS)
        if (frameCount % 5 === 0) {
            try {
                // Create temp canvas for API (without boxes)
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = canvas.width;
                tempCanvas.height = canvas.height;
                const tempCtx = tempCanvas.getContext('2d');
                tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
                
                const blob = await new Promise(resolve => tempCanvas.toBlob(resolve, 'image/jpeg', 0.7));
                
                // Send to API
                const formData = new FormData();
                formData.append('file', blob, 'frame.jpg');
                
                const result = await fetch(`${API_BASE_URL}/predict/image?confidence=${confidenceThreshold}&draw_boxes=false`, {
                    method: 'POST',
                    body: formData
                });
                
                const data = await result.json();
                
                if (data.success) {
                    lastDetections = data.detections || [];
                    if (lastDetections.length > 0) {
                        updateStats(data.detections);
                        addToLog(data.detections);
                    }
                    handleSortingDecision(data.sorting_decision);
                }
                
            } catch (error) {
                console.error('Camera frame processing error:', error);
            }
        }
        
        // Continue processing
        animationFrameId = requestAnimationFrame(processFrame);
    }
    
    processFrame();
}

// ==================== DRAWING FUNCTIONS ====================

function resizeCanvas() {
    const canvas = document.getElementById('detectionCanvas');
    const img = document.getElementById('detectionImage');
    const videoDisplay = document.getElementById('uploadedVideoDisplay');
    
    if (img && img.complete && img.naturalWidth > 0) {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
    } else if (videoDisplay && videoDisplay.videoWidth > 0) {
        canvas.width = videoDisplay.videoWidth;
        canvas.height = videoDisplay.videoHeight;
    }
}

function drawDetectionsOnImage(detections, img) {
    const canvas = document.getElementById('detectionCanvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    canvas.style.display = 'block';
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const filteredDetections = detections.filter(d => classFilters[d.class_name]);
    filteredDetections.forEach(det => drawBoundingBox(ctx, det));
}

function drawDetectionsOnCanvas(detections, ctx, width, height) {
    const filteredDetections = detections.filter(d => classFilters[d.class_name]);
    filteredDetections.forEach(det => drawBoundingBox(ctx, det));
}

function drawBoundingBox(ctx, detection) {
    const { bbox, class_name, confidence, category } = detection;
    const { x1, y1, x2, y2 } = bbox;
    
    // Set color based on category
    const color = category === 'Inorganic' ? '#00ff00' : '#ff6600';
    
    // Draw box
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    
    // Draw label background
    const label = `${class_name.replace('_', ' ')} (${category}) ${Math.round(confidence * 100)}%`;
    ctx.font = 'bold 14px Arial';
    const textWidth = ctx.measureText(label).width;
    
    ctx.fillStyle = color;
    ctx.fillRect(x1, y1 - 25, textWidth + 10, 25);
    
    // Draw label text
    ctx.fillStyle = '#000';
    ctx.fillText(label, x1 + 5, y1 - 7);
}

// ==================== SORTING DECISION ====================

function handleSortingDecision(decision) {
    if (!decision) return;
    
    let decisionText = 'WAITING';
    
    switch(decision.signal) {
        case 'GREEN':
            decisionText = 'INORGANIC STREAM';
            break;
        case 'RED':
            decisionText = 'ORGANIC STREAM';
            break;
        case 'MIXED':
            decisionText = 'SEPARATE STREAMS';
            break;
        case 'IDLE':
            decisionText = 'NO DETECTION';
            break;
    }
    
    updateSortingDecision(decisionText);
    updateSortingSignal(decision.signal);
}

function updateSortingDecision(decision) {
    const topDecision = document.getElementById('sortingDecisionTop');
    const bottomDecision = document.getElementById('sortingDecisionBottom');
    
    if (topDecision) {
        topDecision.querySelector('.decision-text').textContent = `Sorting Decision: ${decision}`;
    }
    if (bottomDecision) {
        bottomDecision.querySelector('.decision-text').textContent = `Sorting Decision: ${decision}`;
    }
}

function updateSortingSignal(signal) {
    const greenLight = document.getElementById('signalGreen');
    const redLight = document.getElementById('signalRed');
    
    if (!greenLight || !redLight) return;
    
    greenLight.classList.remove('active');
    redLight.classList.remove('active');
    
    switch(signal) {
        case 'GREEN':
            greenLight.classList.add('active');
            break;
        case 'RED':
            redLight.classList.add('active');
            break;
        case 'MIXED':
            greenLight.classList.add('active');
            redLight.classList.add('active');
            break;
    }
}

// ==================== STATISTICS ====================

function updateStats(detections) {
    detections.forEach(det => {
        if (!classFilters[det.class_name]) return;
        
        stats.total++;
        if (stats.byClass[det.class_name] !== undefined) {
            stats.byClass[det.class_name]++;
        }
        
        if (det.category === 'Inorganic') {
            stats.inorganic++;
        } else {
            stats.organic++;
        }
    });
    
    // Update UI
    document.getElementById('totalItems').textContent = stats.total;
    document.getElementById('inorganicCount').textContent = stats.inorganic;
    document.getElementById('organicCount').textContent = stats.organic;
    
    // Update charts
    if (statsChart) {
        statsChart.data.datasets[0].data = [stats.inorganic, stats.organic];
        statsChart.update('none');
    }
    
    if (pieChart) {
        pieChart.data.datasets[0].data = [
            stats.byClass.bottle,
            stats.byClass.can,
            stats.byClass.bag,
            stats.byClass.banana_peel,
            stats.byClass.eggshell,
            stats.byClass.leaves
        ];
        pieChart.update('none');
    }
}

// ==================== LOGGING ====================

function addToLog(detections) {
    const now = new Date();
    const timestamp = now.toLocaleString('vi-VN');
    
    detections.forEach(det => {
        if (!classFilters[det.class_name]) return;
        
        const log = {
            timestamp,
            class_name: det.class_name,
            category: det.category,
            confidence: det.confidence
        };
        
        detectionLogs.unshift(log);
    });
    
    // Keep only last 100 logs
    if (detectionLogs.length > 100) {
        detectionLogs = detectionLogs.slice(0, 100);
    }
    
    renderLogs();
}

function renderLogs() {
    const logTable = document.getElementById('logTable');
    if (!logTable) return;
    
    const searchTerm = document.getElementById('logSearch')?.value?.toLowerCase() || '';
    
    const filteredLogs = detectionLogs.filter(log => 
        log.class_name.toLowerCase().includes(searchTerm) ||
        log.category.toLowerCase().includes(searchTerm) ||
        log.timestamp.includes(searchTerm)
    );
    
    if (filteredLogs.length === 0) {
        logTable.innerHTML = '<div class="log-placeholder">No detections yet</div>';
        return;
    }
    
    logTable.innerHTML = filteredLogs.slice(0, 50).map(log => `
        <div class="log-entry">
            <span class="log-time">${log.timestamp}</span>
            <span class="log-class ${log.category.toLowerCase()}">${log.class_name.replace('_', ' ')} (${log.category})</span>
        </div>
    `).join('');
}

function filterLogs() {
    renderLogs();
}

// ==================== UI CONTROLS ====================

function updateConfidence(value) {
    confidenceThreshold = value / 100;
    document.getElementById('confidenceValue').textContent = confidenceThreshold.toFixed(2);
}

function updateClassFilter() {
    const checkboxes = document.querySelectorAll('.class-filters input[type="checkbox"]');
    checkboxes.forEach(cb => {
        classFilters[cb.dataset.class] = cb.checked;
    });
    
    // Only re-draw if we're in image mode and have detections
    if (currentSource === 'image' || currentSource === 'upload') {
        if (lastImageDetections.length > 0 && originalImageSrc) {
            const img = document.getElementById('detectionImage');
            
            // Create a temporary image to ensure original is loaded
            const tempImg = new Image();
            tempImg.onload = function() {
                drawImageWithDetections(tempImg, lastImageDetections);
            };
            tempImg.src = originalImageSrc;
        }
    }
    // For video/camera, the filter is applied in real-time during processing
    
    console.log('Class filters updated:', classFilters);
}

// ==================== SNAPSHOT ====================

function takeSnapshot() {
    const canvas = document.getElementById('detectionCanvas');
    const img = document.getElementById('detectionImage');
    const videoDisplay = document.getElementById('uploadedVideoDisplay');
    const cameraVideo = document.getElementById('videoElement');
    
    // Create snapshot canvas
    const snapshotCanvas = document.createElement('canvas');
    const ctx = snapshotCanvas.getContext('2d');
    
    // Determine source
    let source = null;
    if (currentSource === 'camera' && cameraVideo && cameraVideo.srcObject) {
        source = cameraVideo;
        snapshotCanvas.width = cameraVideo.videoWidth || 640;
        snapshotCanvas.height = cameraVideo.videoHeight || 480;
    } else if (currentSource === 'video' && videoDisplay && videoDisplay.src) {
        source = videoDisplay;
        snapshotCanvas.width = videoDisplay.videoWidth || 640;
        snapshotCanvas.height = videoDisplay.videoHeight || 480;
    } else if (img && img.src && img.src !== window.location.href) {
        source = img;
        snapshotCanvas.width = img.naturalWidth || 640;
        snapshotCanvas.height = img.naturalHeight || 480;
    }
    
    if (!source) {
        alert('No content to snapshot. Please upload an image/video or start camera first.');
        return;
    }
    
    // Draw source
    ctx.drawImage(source, 0, 0, snapshotCanvas.width, snapshotCanvas.height);
    
    // Draw detections overlay if exists
    if (canvas && canvas.width > 0 && canvas.height > 0) {
        ctx.drawImage(canvas, 0, 0, snapshotCanvas.width, snapshotCanvas.height);
    }
    
    // Download
    snapshotCanvas.toBlob(blob => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `ecosort_snapshot_${Date.now()}.png`;
        a.click();
        URL.revokeObjectURL(url);
        console.log('📸 Snapshot saved');
    }, 'image/png');
}

// ==================== EXPORT FUNCTIONS ====================

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
        let csv = 'Timestamp,Class,Category,Confidence\n';
        detectionLogs.forEach(log => {
            csv += `"${log.timestamp}","${log.class_name}","${log.category}",${log.confidence}\n`;
        });
        const blob = new Blob([csv], { type: 'text/csv' });
        downloadBlob(blob, 'detection_logs.csv');
    }
    closeModal();
}

async function downloadExcel() {
    try {
        window.open(`${API_BASE_URL}/logs/export/excel`, '_blank');
    } catch (error) {
        alert('Excel export requires backend server. Please use CSV export.');
    }
    closeModal();
}

function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

// ==================== NAVIGATION ====================

function logout() {
    if (confirm('Are you sure you want to logout?')) {
        stopCamera();
        stopClassification();
        location.reload();
    }
}

// Page navigation
document.querySelectorAll('.nav-links a[data-page]').forEach(link => {
    link.addEventListener('click', function(e) {
        e.preventDefault();
        document.querySelectorAll('.nav-links a').forEach(l => l.classList.remove('active'));
        this.classList.add('active');
    });
});
