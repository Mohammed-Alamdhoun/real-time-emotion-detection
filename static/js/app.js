let stream = null;
let captureInterval = null;
let isCameraOn = false;
let isPredicting = false;
let isRecording = false;
let currentSessionId = null;
let detectionCount = 0;
let emotionDistribution = {
    angry: 0, contempt: 0, disgust: 0, fear: 0,
    happy: 0, neutral: 0, sad: 0, suprise: 0
};

const video = document.getElementById('video');
const predictionOverlay = document.getElementById('predictionOverlay');
const overlayCtx = predictionOverlay.getContext('2d');
const cameraOff = document.getElementById('cameraOff');
const activeIndicator = document.getElementById('activeIndicator');
const emotionDisplay = document.getElementById('emotionDisplay');
const emotionVisual = document.getElementById('emotionVisual');
const statusDisplay = document.getElementById('statusDisplay');
const cameraDisplay = document.getElementById('cameraDisplay');
const predictionDisplay = document.getElementById('predictionDisplay');
const errorAlert = document.getElementById('errorAlert');
const errorMessage = document.getElementById('errorMessage');
const detectionCountEl = document.getElementById('detectionCount');
const recordingDisplay = document.getElementById('recordingDisplay');
const helpModal = document.getElementById('helpModal');
const startSystemBtn = document.getElementById('startSystemBtn');
const stopSystemBtn = document.getElementById('stopSystemBtn');
const startRecordingBtn = document.getElementById('startRecordingBtn');
const stopRecordingBtn = document.getElementById('stopRecordingBtn');
const helpBtn = document.getElementById('helpBtn');
const closeHelpModal = document.getElementById('closeHelpModal');

const emotionColors = {
    angry: { text: 'text-red-400', bg: 'bg-red-400', icon: 'fa-angry' },
    contempt: { text: 'text-orange-400', bg: 'bg-orange-400', icon: 'fa-meh-rolling-eyes' },
    disgust: { text: 'text-green-400', bg: 'bg-green-400', icon: 'fa-grimace' },
    fear: { text: 'text-purple-400', bg: 'bg-purple-400', icon: 'fa-surprise' },
    happy: { text: 'text-yellow-400', bg: 'bg-yellow-400', icon: 'fa-smile' },
    neutral: { text: 'text-gray-400', bg: 'bg-gray-400', icon: 'fa-meh' },
    sad: { text: 'text-blue-400', bg: 'bg-blue-400', icon: 'fa-sad-tear' },
    suprise: { text: 'text-pink-400', bg: 'bg-pink-400', icon: 'fa-surprise' }
};

function initParticles() {
    const container = document.getElementById('particles-container');
    for (let i = 0; i < 30; i++) {
        const particle = document.createElement('div');
        particle.classList.add('particle');

        const size = Math.random() * 6 + 2;
        particle.style.width = `${size}px`;
        particle.style.height = `${size}px`;
        particle.style.left = `${Math.random() * 100}vw`;
        particle.style.top = `${Math.random() * 100}vh`;

        const animationDuration = Math.random() * 20 + 10;
        particle.style.animation = `float ${animationDuration}s linear infinite`;

        container.appendChild(particle);
    }
}

function createEmotionWave(x, y, color) {
    const wavesContainer = document.getElementById('emotionWaves');
    const wave = document.createElement('div');
    wave.classList.add('emotion-wave');
    wave.style.left = `${x}px`;
    wave.style.top = `${y}px`;
    wave.style.borderColor = color;
    wave.style.width = '0px';
    wave.style.height = '0px';

    wavesContainer.appendChild(wave);

    setTimeout(() => {
        wave.remove();
    }, 2000);
}

function showError(message) {
    errorMessage.textContent = message;
    errorAlert.classList.remove('hidden');
    setTimeout(() => {
        errorAlert.classList.add('hidden');
    }, 5000);
}

function updateEmotionDisplay(emotion, confidence = 0.8) {
    emotionDisplay.textContent = emotion.charAt(0).toUpperCase() + emotion.slice(1);
    emotionDisplay.className = `text-4xl font-bold mb-2 transition-all duration-500 ${emotionColors[emotion].text}`;

    emotionVisual.innerHTML = `<i class="fas ${emotionColors[emotion].icon} text-white"></i>`;

    detectionCount++;
    if (detectionCountEl) {
        detectionCountEl.textContent = detectionCount;
    }
}

async function startSystem() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 }
        });

        video.srcObject = stream;
        isCameraOn = true;
        isPredicting = true;

        cameraOff.classList.add('hidden');
        video.classList.remove('hidden');
        predictionOverlay.classList.remove('hidden');
        activeIndicator.classList.remove('hidden');
        activeIndicator.classList.add('flex');

        startSystemBtn.classList.add('hidden');
        stopSystemBtn.classList.remove('hidden');
        stopSystemBtn.classList.add('flex');

        cameraDisplay.textContent = 'Active';
        cameraDisplay.classList.remove('text-gray-400');
        cameraDisplay.classList.add('text-emerald-400');

        predictionDisplay.textContent = 'Active';
        predictionDisplay.classList.remove('text-gray-400');
        predictionDisplay.classList.add('text-blue-400');

        statusDisplay.textContent = 'Predicting';
        statusDisplay.classList.remove('text-gray-400');
        statusDisplay.classList.add('text-blue-400');

        document.getElementById('welcomeMessage').classList.add('hidden');

        video.addEventListener('loadedmetadata', () => {
            predictionOverlay.width = video.videoWidth;
            predictionOverlay.height = video.videoHeight;
        });

        captureInterval = setInterval(captureFrame, 1000);
    } catch (err) {
        showError('Failed to access camera. Please grant permission.');
        console.error('Camera error:', err);
    }
}

let isProcessing = false;
let consecutiveErrors = 0;
const MAX_CONSECUTIVE_ERRORS = 5;

async function captureFrame() {
    if (isProcessing) {
        console.log('Skipping frame - already processing');
        return;
    }

    if (!video || !predictionOverlay || video.readyState !== 4) {
        console.log('Video not ready');
        return;
    }

    isProcessing = true;

    try {
        const startTime = Date.now();

        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = video.videoWidth;
        tempCanvas.height = video.videoHeight;
        tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);

        const dataURL = tempCanvas.toDataURL('image/jpeg', 0.7);

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 8000);

        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: dataURL }),
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}`);
        }

        const data = await response.json();
        const processingTime = Date.now() - startTime;

        console.log(`Request completed in ${processingTime}ms`);

        overlayCtx.clearRect(0, 0, predictionOverlay.width, predictionOverlay.height);

        if (data.error) {
            updateEmotionDisplay('neutral', 0.1);
            consecutiveErrors++;
            return;
        }

        consecutiveErrors = 0;
        if (consecutiveErrors === 0) {
            clearInterval(captureInterval);
            captureInterval = setInterval(captureFrame, 1000);
        }

        const [x, y, x2, y2] = data.box;

        overlayCtx.strokeStyle = '#10b981';
        overlayCtx.lineWidth = 3;
        overlayCtx.shadowColor = '#10b981';
        overlayCtx.shadowBlur = 10;
        overlayCtx.strokeRect(x, y, x2 - x, y2 - y);

        overlayCtx.shadowBlur = 0;
        overlayCtx.fillStyle = 'rgba(16, 185, 129, 0.9)';
        const labelY = y > 30 ? y - 30 : y2 + 5;
        overlayCtx.fillRect(x, labelY, 150, 25);

        overlayCtx.fillStyle = '#ffffff';
        overlayCtx.font = 'bold 16px Inter, sans-serif';
        overlayCtx.fillText(data.prediction.toUpperCase(), x + 10, labelY + 18);

        const centerX = x + (x2 - x) / 2;
        const centerY = y + (y2 - y) / 2;
        createEmotionWave(centerX, centerY, '#10b981');

        updateEmotionDisplay(data.prediction, 0.8);

    } catch (err) {
        consecutiveErrors++;
        console.error('Prediction error:', err);

        if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
            showError(`Server connection issues. ${consecutiveErrors} consecutive failures.`);

            if (captureInterval) {
                clearInterval(captureInterval);
                captureInterval = setInterval(captureFrame, 3000);
                showError('Reduced prediction frequency due to connection issues');
            }
        } else if (err.name === 'AbortError') {
            console.log('Request timeout');
        } else {
            console.log('Connection failed, retrying...');
        }
    } finally {
        isProcessing = false;
    }
}

async function startRecording() {
    if (!isPredicting) {
        showError('Please start the system first before recording');
        return;
    }

    try {
        const response = await fetch('/start_session', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        const data = await response.json();

        if (response.ok) {
            isRecording = true;
            currentSessionId = data.session_id;

            recordingDisplay.textContent = 'Active';
            recordingDisplay.classList.remove('text-gray-400');
            recordingDisplay.classList.add('text-purple-400');

            startRecordingBtn.classList.add('hidden');
            stopRecordingBtn.classList.remove('hidden');
            stopRecordingBtn.classList.add('flex');

            showError(`Recording started: ${data.session_id}`);
            setTimeout(() => errorAlert.classList.add('hidden'), 3000);
            errorAlert.classList.remove('bg-red-500/10', 'border-red-500/20');
            errorAlert.classList.add('bg-green-500/10', 'border-green-500/20');
            errorMessage.classList.remove('text-red-400');
            errorMessage.classList.add('text-green-400');
        } else {
            showError(data.error || 'Failed to start recording');
        }
    } catch (err) {
        console.error('Recording error:', err);
        showError('Failed to start recording session');
    }
}

async function stopRecording() {
    if (!isRecording) {
        return;
    }

    try {
        const response = await fetch('/stop_session', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        const data = await response.json();

        if (response.ok) {
            isRecording = false;

            recordingDisplay.textContent = 'Inactive';
            recordingDisplay.classList.remove('text-purple-400');
            recordingDisplay.classList.add('text-gray-400');

            stopRecordingBtn.classList.add('hidden');
            startRecordingBtn.classList.remove('hidden');

            showError(`Recording stopped: ${data.message}`);
            setTimeout(() => errorAlert.classList.add('hidden'), 3000);
            errorAlert.classList.remove('bg-red-500/10', 'border-red-500/20');
            errorAlert.classList.add('bg-green-500/10', 'border-green-500/20');
            errorMessage.classList.remove('text-red-400');
            errorMessage.classList.add('text-green-400');

            currentSessionId = null;
        } else {
            showError(data.error || 'Failed to stop recording');
        }
    } catch (err) {
        console.error('Stop recording error:', err);
        showError('Failed to stop recording session');
    }
}

function stopSystem() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    if (captureInterval) {
        clearInterval(captureInterval);
        captureInterval = null;
    }

    video.srcObject = null;
    isCameraOn = false;
    isPredicting = false;

    if (isRecording) {
        stopRecording();
    }

    cameraOff.classList.remove('hidden');
    video.classList.add('hidden');
    predictionOverlay.classList.add('hidden');
    activeIndicator.classList.add('hidden');

    stopSystemBtn.classList.add('hidden');
    startSystemBtn.classList.remove('hidden');

    cameraDisplay.textContent = 'Inactive';
    cameraDisplay.classList.remove('text-emerald-400');
    cameraDisplay.classList.add('text-gray-400');

    predictionDisplay.textContent = 'Inactive';
    predictionDisplay.classList.remove('text-blue-400');
    predictionDisplay.classList.add('text-gray-400');

    statusDisplay.textContent = 'Idle';
    statusDisplay.classList.remove('text-blue-400');
    statusDisplay.classList.add('text-gray-400');
}

startSystemBtn.addEventListener('click', startSystem);
stopSystemBtn.addEventListener('click', stopSystem);
startRecordingBtn.addEventListener('click', startRecording);
stopRecordingBtn.addEventListener('click', stopRecording);
helpBtn.addEventListener('click', () => helpModal.classList.remove('hidden'));
closeHelpModal.addEventListener('click', () => helpModal.classList.add('hidden'));

initParticles();

window.addEventListener('beforeunload', () => {
    if (captureInterval) {
        clearInterval(captureInterval);
    }
    if (isRecording) {
        stopRecording();
    }
    stopSystem();
});
