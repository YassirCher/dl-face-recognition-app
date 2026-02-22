document.addEventListener('DOMContentLoaded', () => {
    const backboneSelect = document.getElementById('backbone-select');
    const classifierSelect = document.getElementById('classifier-select');
    const switchBtn = document.getElementById('switch-btn');
    const btnText = document.getElementById('btn-text');
    const btnLoader = document.getElementById('btn-loader');
    const currentBackboneSpan = document.getElementById('current-backbone');
    const currentClassifierSpan = document.getElementById('current-classifier');
    const statusSpan = document.getElementById('system-status');
    const modelForm = document.getElementById('model-form');

    // Video Upload Elements
    const videoUploadInput = document.getElementById('video-upload');
    const uploadBtn = document.getElementById('upload-btn');
    const uploadText = document.getElementById('upload-text');
    const uploadLoader = document.getElementById('upload-loader');

    let modelsData = {};

    // Only fetch models if we are on a page with the controls
    if (backboneSelect) {
        fetch('/api/models')
            .then(response => response.json())
            .then(data => {
                modelsData = data;
                populateBackbones();
            })
            .catch(err => console.error('Error fetching models:', err));
    }

    // Always fetch current model status for display checks if elements exist
    if (currentBackboneSpan) {
        fetch('/api/current_model')
            .then(response => response.json())
            .then(data => {
                updateStatusDisplay(data.backbone, data.classifier);
            })
            .catch(err => console.error('Error fetching current model:', err));
    }

    function populateBackbones() {
        if (!backboneSelect) return;
        backboneSelect.innerHTML = '<option value="" disabled selected>Select Backbone</option>';
        modelsData.backbones.forEach(bb => {
            const option = document.createElement('option');
            option.value = bb;
            option.textContent = bb;
            backboneSelect.appendChild(option);
        });
    }

    if (backboneSelect) {
        backboneSelect.addEventListener('change', () => {
            const selectedBackbone = backboneSelect.value;
            if (!classifierSelect) return;

            classifierSelect.innerHTML = '<option value="" disabled selected>Select Classifier</option>';

            if (selectedBackbone && modelsData.classifiers[selectedBackbone]) {
                modelsData.classifiers[selectedBackbone].forEach(cl => {
                    const option = document.createElement('option');
                    option.value = cl;
                    option.textContent = cl;
                    classifierSelect.appendChild(option);
                });
                classifierSelect.disabled = false;
            } else {
                classifierSelect.disabled = true;
            }
        });
    }

    if (modelForm) {
        modelForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const backbone = backboneSelect.value;
            const classifier = classifierSelect.value;

            if (!backbone || !classifier) return;

            // UI Loading State
            setLoading(true);

            fetch('/api/set_model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ backbone, classifier }),
            })
                .then(response => response.json())
                .then(data => {
                    setLoading(false);
                    if (data.success) {
                        updateStatusDisplay(backbone, classifier);
                        showToast(data.message, 'success');
                    } else {
                        showToast(data.error, 'error');
                    }
                })
                .catch(err => {
                    setLoading(false);
                    showToast('Failed to switch model', 'error');
                    console.error(err);
                });
        });
    }

    // Video Upload Logic
    if (uploadBtn && videoUploadInput) {
        uploadBtn.addEventListener('click', () => {
            const file = videoUploadInput.files[0];
            if (!file) {
                showToast('Please select a video file first', 'error');
                return;
            }

            setUploadLoading(true);
            const formData = new FormData();
            formData.append('video', file);

            fetch('/api/upload_video', {
                method: 'POST',
                body: formData
            })
                .then(response => response.json())
                .then(data => {
                    setUploadLoading(false);
                    if (data.success) {
                        showToast(data.message, 'success');
                        // Optional: Reload video feed image to force refresh if browser caches
                        const videoImg = document.querySelector('img[src*="video_feed"]');
                        if (videoImg) {
                            videoImg.src = videoImg.src.split('?')[0] + '?t=' + new Date().getTime();
                        }
                    } else {
                        showToast(data.error, 'error');
                    }
                })
                .catch(err => {
                    setUploadLoading(false);
                    showToast('Upload failed', 'error');
                    console.error(err);
                });
        });
    }

    function setLoading(isLoading) {
        if (!switchBtn || !btnText || !btnLoader) return;
        switchBtn.disabled = isLoading;
        if (isLoading) {
            btnText.textContent = 'Loading Model...';
            btnLoader.classList.remove('hidden');
        } else {
            btnText.textContent = 'Apply Configuration';
            btnLoader.classList.add('hidden');
        }
    }

    function setUploadLoading(isLoading) {
        if (!uploadBtn || !uploadText || !uploadLoader) return;
        uploadBtn.disabled = isLoading;
        if (isLoading) {
            uploadText.textContent = 'Uploading...';
            uploadLoader.classList.remove('hidden');
        } else {
            uploadText.textContent = 'Upload & Play';
            uploadLoader.classList.add('hidden');
        }
    }

    function updateStatusDisplay(backbone, classifier) {
        if (!currentBackboneSpan || !currentClassifierSpan || !statusSpan) return;

        currentBackboneSpan.textContent = backbone || 'None';
        currentClassifierSpan.textContent = classifier || 'None';

        if (backbone && classifier) {
            statusSpan.innerHTML = '<i class="fa-solid fa-check-circle mr-1"></i> Active';
            statusSpan.className = "text-green-400 text-sm font-semibold";
        } else {
            statusSpan.innerHTML = '<i class="fa-solid fa-pause-circle mr-1"></i> Idle';
            statusSpan.className = "text-yellow-400 text-sm font-semibold";
        }
    }

    // Simple toast notification
    function showToast(message, type) {
        // Reuse btnText if available, or create toast
        if (btnText && !btnText.textContent.includes('Loading')) {
            const originalText = btnText.textContent;
            btnText.textContent = message;
            // Basic color feedback
            if (switchBtn) {
                const originalColor = switchBtn.className;
                switchBtn.className = type === 'success'
                    ? "w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-green-600 transition-all"
                    : "w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-red-600 transition-all";

                setTimeout(() => {
                    btnText.textContent = originalText;
                    switchBtn.className = originalColor;
                }, 3000);
            }
        } else if (uploadText && !uploadText.textContent.includes('Uploading')) {
            const originalText = uploadText.textContent;
            uploadText.textContent = message;
            setTimeout(() => {
                uploadText.textContent = originalText;
            }, 3000);
        }
        else {
            console.log(type.toUpperCase() + ": " + message);
        }
    }
});
