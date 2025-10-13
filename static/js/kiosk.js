window.errorLog = [];
window.debug = false;
// Variable to track if models are loaded
window.faceApiModelsLoaded = false;

function dlog(text) {
    if (window.debug) {
        console.log(text);
    }
}

function isMobileDevice() {
    // Check for touch capability and typical mobile user agents
    const hasTouchCapability = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
    const userAgent = navigator.userAgent.toLowerCase();
    
    // Common mobile device patterns
    const mobileRegex = /android|webos|iphone|ipad|ipod|blackberry|iemobile|opera mini|mobile|tablet/i;
    
    // Check if device is mobile based on user agent OR touch capabilities with small screen
    if (mobileRegex.test(userAgent)) {
        return true;
    }
    
    // Additional check for touch devices with small screen (hybrid detection)
    if (hasTouchCapability && window.innerWidth <= 1024) {
        return true;
    }
    
    return false;
}

function enterFullscreenForDocument() {
    const elem = document.documentElement; // Targets the <html> element
    if (elem.requestFullscreen) {
      elem.requestFullscreen();
    } else if (elem.webkitRequestFullscreen) { /* Safari */
      elem.webkitRequestFullscreen();
    } else if (elem.msRequestFullscreen) { /* IE11 */
      elem.msRequestFullscreen();
    }
    if (!isMobileDevice()) {
    setInterval(function() {        
          if (elem.requestFullscreen) {
            elem.requestFullscreen();
          } else if (elem.webkitRequestFullscreen) { /* Safari */
            elem.webkitRequestFullscreen();
          } else if (elem.msRequestFullscreen) { /* IE11 */
            elem.msRequestFullscreen();
          }
      }, 1000);
    }
}

const okBEEP = new Audio('/static/resource/okBEEP.mp3');
const failBEEP = new Audio('/static/resource/failBEEP.mp3');

// Variables for timer
let inactivityTimer;
let countdown;
const timeoutDuration = 15000; // 15 seconds

function renderLoader() {
            const loaderslot = document.getElementById("loaderSlot")
            loaderslot.insertAdjacentHTML("afterbegin", '<div class="loader"></div>')
        }

function removeLoader() {
            const loaderslot = document.getElementById("loaderSlot")
            const loader = loaderslot.querySelector(".loader")
            if (loader) {
                loader.remove()
            }
        }
 
window.dPopupOpen = false;
// Function to open the popup
function openDPopup() {
    window.dPopupOpen = true;
     document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape' && window.dPopupOpen) {
            closeDPopup();
        }
        if (event.key === 'Enter' && window.dPopupOpen) {
            document.getElementById('destination-submit-button').click();
        }
        if (event.key === 'ArrowUp' && window.dPopupOpen) {
            var select = document.getElementById('destinationSelect');
            select.selectedIndex = (select.selectedIndex > 0) ? select.selectedIndex - 1 : select.options.length - 1;
            var event = new Event('change');
            select.dispatchEvent(event);
        }
        if (event.key === 'ArrowDown' && window.dPopupOpen) {
            var select = document.getElementById('destinationSelect');
            select.selectedIndex = (select.selectedIndex < select.options.length - 1) ? select.selectedIndex + 1 : 0;
            var event = new Event('change');
            select.dispatchEvent(event);
        }
    });
    var popupOverlay = document.getElementById('popupOverlay');
    var destinationSelect = document.getElementById('destinationSelect');
    popupOverlay.classList.add('active');
    startInactivityTimer();
    // Add event listeners for user activity
    document.addEventListener('keypress', resetInactivityTimer);
    destinationSelect.addEventListener('change', resetInactivityTimer);
}
 
// Function to close the popup
function closeDPopup() {
    window.dPopupOpen = false;
    // Clear the key event listeners
    document.onkeydown = null;
    var popupOverlay = document.getElementById('popupOverlay');
    var destinationSelect = document.getElementById('destinationSelect');
    popupOverlay.classList.remove('active')
    clearTimeout(inactivityTimer);
    clearInterval(countdown);
    // Remove event listeners
    document.removeEventListener('keypress', resetInactivityTimer);
    destinationSelect.removeEventListener('change', resetInactivityTimer);
}
 
// Function to start the inactivity timer
function startInactivityTimer() {
    var countdownElement = document.getElementById('countdown');
    var timerProgress = document.querySelector('.timer-progress');
    // Clear any existing timer
    clearTimeout(inactivityTimer);
    clearInterval(countdown);
    
    // Reset countdown UI
    let timeLeft = timeoutDuration / 1000;
    countdownElement.textContent = timeLeft;
    timerProgress.style.width = '100%';
    
    // Start countdown animation
    countdown = setInterval(function() {
        timeLeft--;
        countdownElement.textContent = timeLeft;
        timerProgress.style.width = (timeLeft / (timeoutDuration / 1000)) * 100 + '%';
    }, 1000);
    
    // Set timeout to close popup
    inactivityTimer = setTimeout(function() {
        removeLoader()
        closeDPopup();
    }, timeoutDuration);
}
 
// Function to reset the inactivity timer on user activity
function resetInactivityTimer() {
    startInactivityTimer();
}

async function getUserLocation() {
    try {
        const response = await fetch("/api/getUserLocation", {
            method: 'POST',
            body: JSON.stringify({}),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        const responseJson = await response.json();

        switch (responseJson.status) {
            case "error":
                failBEEP.play();
                createAlertPopup(5000, null, 'Error While Getting User Location', responseJson.errorinfo)
                return 'error'
            case "ok":
                return responseJson.location
            default:
                return null
        }
        
    } catch (error) {
        failBEEP.play();
        removeLoader()
        window.errorLog.push(error)
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        return null
    }
}

async function getLocationId(type) {
    try {
        const response = await fetch("/api/getLocationId", {
            method: 'POST',
            body: JSON.stringify({
                "type": type
            }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        const responseJson = await response.json();

        switch (responseJson.status) {
            case "error":
                failBEEP.play();
                createAlertPopup(5000, null, 'Error While Getting Location Info', responseJson.errorinfo)
                return 'error'
                break;
            case "ok":
                if (type == 1) {
                    window.destinationLocationJson = responseJson.locationJson
                } else if (type == 2) {
                    window.floorLocationJson = responseJson.locationJson
                }
                return responseJson.locationJson
                break;
            default:
                return None
        }
        
    } catch (error) {
        failBEEP.play();
        removeLoader()
        window.errorLog.push(error)
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        return 0
    }
}

function createAlertPopup(closetimeout, type = null, title, body, alertid = '') {
    let alertContainer = document.getElementById('alertContainer');
    let alertElement = document.createElement('div');
    alertElement.setAttribute('id', alertid);
    alertElement.classList.add('alert');
    dlog(type);
    if (type != null && type != undefined && type != '') {
        alertElement.classList.add(type);
    }

    let closeButton = document.createElement('span');
    closeButton.classList.add('closebtn');
    closeButton.setAttribute('onclick', "this.parentElement.remove()");
    closeButton.innerHTML = '&times;';

    let titleText = document.createElement('strong');
    titleText.textContent = title;

    let bodyText = document.createElement('p');
    bodyText.innerHTML = body;

    alertElement.appendChild(closeButton);
    alertElement.appendChild(titleText);
    alertElement.appendChild(bodyText);

    alertContainer.appendChild(alertElement);

    if (closetimeout != null) {
        let closeTimout = setTimeout(function () { alertElement.remove(); }, closetimeout);
    }
}

async function getStudentIdFromCard(cardid) {
    try {
        const response = await fetch('/api/getStudentIdFromCard', {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ cardid: cardid })
        });
        const data = await response.json();
        if (data.status === 'ok') {
            return data.studentid;
        } else {
            failBEEP.play();
            createAlertPopup(5000, null, 'Error', data.errorinfo || 'Unknown error');
            return null;
        }
    } catch (error) {
        failBEEP.play();
        removeLoader()
        window.errorLog.push(error);
        createAlertPopup(5000, null, 'Error', 'Failed to fetch student ID from card');
        return null;
    }
}

async function getStudentImage(studentid) {
    try {
        const response = await fetch("/api/getStudentImage", {
            method: 'POST',
            body: JSON.stringify({ "studentid": studentid }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        const responseJson = await response.json();

        if (responseJson.status === "error") {
            failBEEP.play();
            createAlertPopup(5000, null, 'Error Fetching Student Image', responseJson.errorinfo);
            return 'error';
        }

        return responseJson.studentBase64Image;
    } catch (error) {
        failBEEP.play();
        removeLoader()
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while fetching student image');
        return 'error';
    }
}

async function determineUserLocationType() {
    var userLocation = await getUserLocation();
    var type = 0;

    Object.values(window.destinationLocationJson).forEach(location => {
        if (location[0] == userLocation) {
            type = 1 // Destination
        }
    });
    Object.values(window.floorLocationJson).forEach(location => {
        if (location[0] == userLocation) {
            type = 2 // Floor
        }
    });
    return type; // Unknown
}

async function approvePassByCard(cardid) {
    try {
        try {
            var studentid = await getStudentIdFromCard(cardid)
            var passDestinationID = await getStudents({studentid: studentid, all: true})
            passDestinationID = passDestinationID[0][3]
            var passDestination = window.destinationLocationJson[passDestinationID][0];
            dlog(passDestination)
        } catch (error) {dlog(error)}
        const response = await fetch("/api/approvePassByCard", {
            method: 'POST',
            body: JSON.stringify({
                "cardid": cardid,
                "image": window.faceBase64,
                "kiosk": true
            }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        if (!response.ok) {
            throw new Error('Network error');
            failBEEP.play();
        }
        const responseJson = await response.json();

        var passid = responseJson.passid

        // Fetch student image if available
        let studentImageUrl = '/static/resource/studentImagePlaceholder.png';
        let studentName = '';
        if (responseJson.studentid) {
            // Fetch image from API
            const imgResp = await fetch("/api/getStudentImage", {
                method: 'POST',
                body: JSON.stringify({ studentid: responseJson.studentid }),
                headers: {
                    Accept: 'application/json',
                    'Content-Type': 'application/json',
                },
            });
            if (imgResp.ok) {
                const imgJson = await imgResp.json();
                if (imgJson.status === "ok") {
                    studentImageUrl = imgJson.studentBase64Image;
                }
            }
            // Fetch student name from API
            const nameResp = await fetch("/api/getStudentInfo", {
                method: 'POST',
                body: JSON.stringify({ studentid: responseJson.studentid }),
                headers: {
                    Accept: 'application/json',
                    'Content-Type': 'application/json',
                },
            });
            if (nameResp.ok) {
                const nameJson = await nameResp.json();
                if (nameJson.status === "ok" && Array.isArray(nameJson.studentinfo)) {
                    studentName = nameJson.studentinfo[0];
                }
            }
        }

        // Handle elapsed time and warnings
        let alertType = 'success';
        let elapsedstring = 'The student has been approved.';
        let elapsedTime = responseJson.elapsedtime;

        // Parse elapsed time into hours, minutes, and seconds
        if (elapsedTime && Array.isArray(elapsedTime)) {
            let [hours, minutes, seconds] = elapsedTime;
            let elapsedTimeStr = '';
            if (hours) elapsedTimeStr += `${hours} Hour${hours > 1 ? 's' : ''} `;
            if (minutes) elapsedTimeStr += `${minutes} Minute${minutes > 1 ? 's' : ''} `;
            if (seconds) elapsedTimeStr += `${seconds} Second${seconds > 1 ? 's' : ''}`;
            elapsedTime = elapsedTimeStr.trim();
        } else {
            elapsedTime = '';
        }

        if (responseJson.elapsedtimewarning === 'min') {
            alertType = 'warning';
            elapsedstring = 'The travel time of the student might be too short.'
        } else if (responseJson.elapsedtimewarning === 'warning') {
            alertType = 'warning';
            elapsedstring = 'The student has been traveling for an extended period of time.'
        } else if (responseJson.elapsedtimewarning === 'alert') {
            alertType = '';
            elapsedstring = 'The student has been traveling for too long.'
        }

        if (passDestination == null || passDestination == undefined) {
            var passDestinationID = await getStudents({studentid: responseJson.studentid, all: true})
            passDestinationID = passDestinationID[0][3]
            var passDestination = window.destinationLocationJson[passDestinationID][0];
        }
        var studentFloorName = await getStudentFloorName(responseJson.studentid);

        dlog(studentImageUrl)

        removeLoader()
        if (responseJson.status === "ok") {
            okBEEP.play();
            showApprovalScreen(studentName || 'Unknown', `${passDestination} - ${studentFloorName}`, studentImageUrl || '/static/resource/studentImagePlaceholder.png');
        } else {
            failBEEP.play();
            showFailureScreen(studentName || 'Unknown', `${passDestination} - ${studentFloorName}` || 'Unknown', studentImageUrl || '/static/resource/studentImagePlaceholder.png', responseJson.errorinfo || 'Unknown error');
        }
    } catch (error) {
        failBEEP.play();
        removeLoader()
        window.errorLog.push(error)
        createAlertPopup(5000, null, 'Error', 'Error while approving pass');
    }
}

let video = document.getElementById('video');
let popup = document.getElementById('popup');
let referenceImg = document.getElementById('referenceImg');
let statusText = document.getElementById('status');
let confidenceText = document.getElementById('confidenceText');
let timerElement = document.getElementById('timer');
let stream = null;
let faceDetectionInterval = null;
let referenceDescriptor = null;
let countdownTimer = null;
let recognitionResult = false;
let canvas = null;
let ctx = null;
let frozen = false;
let lastDetection = null;

// Close button event
document.querySelector('.close-button').addEventListener('click', function() {
    closePopup(false);
});

// Function to start face recognition
async function startFaceRecognition(base64Image, studentName) {
    document.getElementById('studentName').textContent = studentName || '';
    // Reset result
    recognitionResult = false;
    frozen = false;
    lastDetection = null;
    
    // Show the popup
    popup.style.display = 'flex';
    
    // Set the reference image
    referenceImg.src = base64Image;
    
    try {
        // iOS Safari compatibility check
        if (typeof navigator.mediaDevices === 'undefined' || !navigator.mediaDevices.getUserMedia) {
            throw new Error("Camera access is not supported in this browser. Please use a different browser or device.");
        }
        
        // If models are already loaded, skip loading
        if (!window.faceApiModelsLoaded) {
            statusText.textContent = "Loading face recognition models...";
            await faceapi.nets.tinyFaceDetector.loadFromUri('/static/face/models');
            await faceapi.nets.faceLandmark68Net.loadFromUri('/static/face/models');
            await faceapi.nets.faceRecognitionNet.loadFromUri('/static/face/models');
            window.faceApiModelsLoaded = true;
        } else {
            statusText.textContent = "Models already loaded, starting recognition...";
        }
        
        // Start camera with selected device
        statusText.textContent = "Accessing camera...";
        const videoConstraints = window.selectedCameraId ? 
            { deviceId: { exact: window.selectedCameraId } } : 
            { facingMode: 'user' };
            
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: videoConstraints
        });
        video.srcObject = stream;
        
        // Setup canvas for drawing face rectangles
        setupFaceDetectionCanvas();
        
        // Process the reference image
        statusText.textContent = "Processing reference image...";
        const referenceImage = await faceapi.fetchImage(base64Image);
        const referenceDetections = await faceapi
            .detectSingleFace(referenceImage, new faceapi.TinyFaceDetectorOptions())
            .withFaceLandmarks()
            .withFaceDescriptor();
        
        if (!referenceDetections) {
            failBEEP.play();
            throw new Error("No face found in reference image");
        }
        
        referenceDescriptor = referenceDetections.descriptor;
        
        // Start the 5-second countdown
        let timeLeft = 5;
        timerElement.textContent = `${timeLeft} seconds remaining`;

        statusText.textContent = "Scanning for matching face...";
        
        // Start face detection interval
        return new Promise((resolve) => {
            // Set up the countdown timer
            countdownTimer = setInterval(() => {
                timeLeft--;
                timerElement.textContent = `${timeLeft} seconds remaining`;
                
                if (timeLeft <= 0) {
                    // Time's up, close the popup and return false
                    clearInterval(countdownTimer);
                    closePopup(false);
                    resolve(false);
                }
            }, 1000);
            
            // Start face detection
            faceDetectionInterval = setInterval(async () => {
                if (frozen) {
                    // If frozen, just redraw the last detection
                    if (lastDetection) {
                        clearCanvas();
                        drawFaceRectangle(lastDetection);
                    }
                    return;
                }
                
                if (video.readyState === video.HAVE_ENOUGH_DATA) {
                    try {
                        // Detect faces in the video
                        const detections = await faceapi
                            .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
                            .withFaceLandmarks()
                            .withFaceDescriptors();
                        
                        // Clear canvas
                        clearCanvas();
                        
                        if (detections.length > 0) {
                            // Find the largest face
                            let largestFace = detections[0];
                            let largestArea = 0;
                            
                            for (const detection of detections) {
                                const area = detection.detection.box.width * detection.detection.box.height;
                                if (area > largestArea) {
                                    largestArea = area;
                                    largestFace = detection;
                                }
                                
                                // Draw rectangle for each detected face
                                drawFaceRectangle(detection.detection);
                            }
                            
                            // Store the last detection for freezing
                            lastDetection = largestFace.detection;
                            
                            // Compare with reference descriptor
                            const distance = faceapi.euclideanDistance(
                                referenceDescriptor, 
                                largestFace.descriptor
                            );
                            
                            // Convert distance to confidence (0-1 scale)
                            // Lower distance = higher similarity
                            let confidence = 1 - Math.min(distance, 1);
                            confidence = Math.max(0, confidence); // Ensure not negative
                            
                            // Update UI with confidence
                            const confidencePercent = Math.round(confidence * 100);
                            confidenceText.textContent = `Match Percentage: ${confidencePercent}%`;
                            
                            // Check if confidence is above 67%
                            if (confidence > 0.57) {
                                frozen = true;
                                // Create temporary canvas to capture image without rectangle
                                const tempCanvas = document.createElement('canvas');
                                tempCanvas.width = video.videoWidth;
                                tempCanvas.height = video.videoHeight;
                                const tempCtx = tempCanvas.getContext('2d');
                                tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
                                window.faceBase64 = tempCanvas.toDataURL('image/png');
                                
                                // Freeze the camera feed by drawing the current frame to canvas
                                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                                // Draw the detection rectangle on top of the frozen frame
                                drawFaceRectangle(lastDetection);
                                // Stop the video stream
                                stream.getTracks().forEach(track => track.stop());
                                // Success! Close the popup and return true
                                clearInterval(countdownTimer);
                                clearInterval(faceDetectionInterval);
                                statusText.textContent = "Success! Verification completed.";
                                statusText.className = "success";
                                
                                // Wait a moment to show success message
                                setTimeout(() => {
                                    renderLoader()
                                    closePopup(true);
                                    resolve(true);
                                }, 750);
                            }
                        } else {
                            confidenceText.textContent = 'Match Percentage: 0% (No faces detected)';
                            lastDetection = null;
                        }
                    } catch (error) {
                        console.error("Face detection error:", error);
                        statusText.textContent = "Error detecting faces";
                    }
                }
            }, 500); // Check every 500ms
        });
        
    } catch (error) {
        failBEEP.play();
        removeLoader()
        console.error("Error:", error);
        statusText.textContent = `Error: ${error.message}`;
        statusText.className = 'error';
        
        // Close the popup after a delay
        setTimeout(() => {
            closePopup(false);
        }, 3000);
        
        return false;
    }
}

async function getStudentInfo(studentid) {
    try {
        const response = await fetch("/api/getStudentInfo", {
            method: 'POST',
            body: JSON.stringify({
                "studentid": studentid
            }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        const responseJson = await response.json();

        switch (responseJson.status) {
            case "error":
                failBEEP.play();
                createAlertPopup(5000, null, 'Error While Getting Student Info', responseJson.errorinfo)
                return 'error'
                break;
            case "ok":
                return responseJson.studentinfo
                break;
            default:
                return None
        }
        
    } catch (error) {
        failBEEP.play();
        removeLoader()
        window.errorLog.push(error)
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        return 0
    }
}

// Setup canvas for drawing face rectangles
function setupFaceDetectionCanvas() {
    // Remove existing canvas if any
    if (canvas) {
        canvas.remove();
    }
    
    // Create canvas element
    canvas = document.createElement('canvas');
    canvas.id = 'faceDetectionCanvas';
    canvas.style.position = 'absolute';
    canvas.style.top = video.offsetTop + 'px';
    canvas.style.left = video.offsetLeft + 'px';
    canvas.style.zIndex = 10;
    
    // Add canvas to video container
    video.parentNode.appendChild(canvas);
    
    // Get context
    ctx = canvas.getContext('2d');
    
    // Set canvas dimensions to match video
    resizeCanvasToVideo();
    
    // Update canvas size when video resizes
    video.addEventListener('resize', resizeCanvasToVideo);
}

// Resize canvas to match video dimensions
function resizeCanvasToVideo() {
    if (canvas && video) {
        canvas.width = video.videoWidth || video.offsetWidth;
        canvas.height = video.videoHeight || video.offsetHeight;
        canvas.style.width = video.offsetWidth + 'px';
        canvas.style.height = video.offsetHeight + 'px';
    }
}

// Clear canvas
function clearCanvas() {
    if (ctx && canvas) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
}

// Draw rectangle around detected face
function drawFaceRectangle(detection) {
    if (!ctx || !canvas) return;
    
    const { x, y, width, height } = detection.box;
    
    // Draw rectangle
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, width, height);
    
    // Draw label background
    ctx.fillStyle = '#00ff00';
    const text = 'Face Detected';
    const textWidth = ctx.measureText(text).width;
    ctx.fillRect(x, y - 20, textWidth + 10, 20);
    
    // Draw label text
    ctx.fillStyle = '#000000';
    ctx.font = '16px Arial';
    ctx.fillText(text, x + 5, y - 5);
}

// Function to close the popup and clean up
function closePopup(result) {
    recognitionResult = result;
    
    if (faceDetectionInterval) {
        clearInterval(faceDetectionInterval);
    }
    
    if (countdownTimer) {
        clearInterval(countdownTimer);
    }
    
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    
    // Remove canvas
    if (canvas) {
        canvas.remove();
        canvas = null;
        ctx = null;
    }
    
    popup.style.display = 'none';
    confidenceText.textContent = 'Match Percentage: 0%';
    statusText.textContent = 'Initializing camera...';
    statusText.className = '';
    timerElement.textContent = '5 seconds remaining';
    
    // Return the result via custom event
    document.dispatchEvent(new CustomEvent('faceRecognitionResult', { detail: result }));
}

async function approveStudent(cardData) {
    let studentId = await getStudentIdFromCard(cardData);
    if (studentId == null) {
        failBEEP.play();
        createAlertPopup(5000, null, 'Error', 'No student ID found for card');
        dlog('No student ID found for card:', cardData);
        return;
    }

    dlog('Student ID from card:', studentId);
    
    let studentImage = await getStudentImage(studentId);
    if (studentImage === 'error') {
        failBEEP.play();
        createAlertPopup(5000, null, 'Error', 'Failed to fetch student image');
        dlog('Error fetching student image for ID:', studentId);
        return;
    }

    dlog('Starting face recognition for student ID:', studentId);

    let studentName = await getStudentInfo(studentId); // Optionally fetch student name if available
    studentName = studentName[0]
    let recognitionSuccess = await startFaceRecognition(studentImage, studentName);
    
    if (recognitionSuccess) {
        let studentPassInfo = await getStudents({studentid: studentId, all: true});
        if (studentPassInfo === 'error' || studentPassInfo.length === 0) {
            return 'NOPASS';
        }
        approvePassByCard(cardData)
    } else {
        failBEEP.play();
        createAlertPopup(5000, 'error', 'Failed', 'Face verification failed or timed out.');
    }
}

async function getStudents(filters) {
    var searchparams = {
        "filter": filters,
    }
    try {
        const response = await fetch("/api/getStudents", {
            method: 'POST',
            body: JSON.stringify(searchparams),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        const responseJson = await response.json();

        switch (responseJson.status) {
            case "error":
                failBEEP.play();
                createAlertPopup(5000, null, 'Error While Fetching Students List', responseJson.errorinfo)
                return 'error'
            case "ok":
                return responseJson.students
            default:
                return None
        }
        
    } catch (error) {
        window.errorLog.push(error)
        dlog('Error:', error);
        failBEEP.play();
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        return 0
    }
}

window.screenShown = false;

function showApprovalScreen(studentName, location, imageUrl) {
    window.screenShown = true;
    // Create overlay element
    const overlay = document.createElement('div');
    overlay.style.position = 'fixed';
    overlay.style.top = '0';
    overlay.style.left = '0';
    overlay.style.width = '100vw';
    overlay.style.height = '100vh';
    overlay.style.backgroundColor = 'green';
    overlay.style.display = 'flex';
    overlay.style.flexDirection = 'column';
    overlay.style.justifyContent = 'center';
    overlay.style.alignItems = 'center';
    overlay.style.zIndex = '9999';
    overlay.style.color = 'white';
    overlay.style.fontFamily = 'Arial, sans-serif';

    // Create text elements
    const approvalText = document.createElement('h1');
    approvalText.textContent = 'Approve Success';
    approvalText.style.margin = '10px';
    approvalText.style.fontSize = '2.5em';

    const nameText = document.createElement('p');
    nameText.textContent = studentName;
    nameText.style.margin = '10px';
    nameText.style.fontSize = '1.8em';

    const locationText = document.createElement('p');
    locationText.textContent = location;
    locationText.style.margin = '10px';
    locationText.style.fontSize = '1.5em';

    // Create image element
    const studentImage = document.createElement('img');
    studentImage.src = imageUrl;
    studentImage.style.margin = '20px';
    studentImage.style.maxWidth = '200px';
    studentImage.style.maxHeight = '200px';
    studentImage.style.borderRadius = '8px';
    studentImage.alt = 'Student Image';

    // Append elements to overlay
    overlay.appendChild(approvalText);
    overlay.appendChild(nameText);
    overlay.appendChild(locationText);
    overlay.appendChild(studentImage);

    // Add to document
    document.body.appendChild(overlay);

    // Optional: Auto-remove after some time (5 seconds)
    setTimeout(() => {
        window.screenShown = false;
        if (document.body.contains(overlay)) {
            document.body.removeChild(overlay);
        }
    }, 5000);
}

async function createNewStudentPass(studentid, floorid, destinationid) {
    try {
        const response = await fetch("/api/newPass", {
            method: 'POST',
            body: JSON.stringify({
                "studentid": studentid,
                "floorid": floorid,
                "destinationid": destinationid
            }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        const responseJson = await response.json();

        removeLoader()

        switch (responseJson.status) {
            case "error":
                failBEEP.play();
                createAlertPopup(5000, null, 'Error Creating Pass', responseJson.errorinfo);
                return 'error';
            case "ok":
                return responseJson.passid;
            default:
                return 'error';
        }

    } catch (error) {
        failBEEP.play();
        removeLoader()
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server');
        return 'error';
    }
}

async function getStudentFloorName(studentid) {
    try {
        const response = await fetch("/api/getStudentFloorName", {
            method: 'POST',
            body: JSON.stringify({ "studentid": studentid }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });
        
        const responseJson = await response.json();
        switch (responseJson.status) {
            case "error":
                createAlertPopup(5000, null, 'Error While Getting Student Floor', responseJson.errorinfo);
                return 'error';
            case "ok":
                return responseJson.floorName;
            default:
                return 'error';
        }
    }
    catch (error) {
        failBEEP.play();
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server');
        return 'error';
    }
}

function showFailureScreen(studentName, location, imageUrl, errorMessage) {
    window.screenShown = true;
    // Create overlay element
    const overlay = document.createElement('div');
    overlay.style.position = 'fixed';
    overlay.style.top = '0';
    overlay.style.left = '0';
    overlay.style.width = '100vw';
    overlay.style.height = '100vh';
    overlay.style.backgroundColor = 'red';
    overlay.style.display = 'flex';
    overlay.style.flexDirection = 'column';
    overlay.style.justifyContent = 'center';
    overlay.style.alignItems = 'center';
    overlay.style.zIndex = '9999';
    overlay.style.color = 'white';
    overlay.style.fontFamily = 'Arial, sans-serif';

    // Create text elements
    const failureText = document.createElement('h1');
    failureText.textContent = 'Approve Failed';
    failureText.style.margin = '10px';
    failureText.style.fontSize = '2.5em';

    const nameText = document.createElement('p');
    nameText.textContent = studentName;
    nameText.style.margin = '10px';
    nameText.style.fontSize = '1.8em';

    const locationText = document.createElement('p');
    locationText.textContent = location;
    locationText.style.margin = '10px';
    locationText.style.fontSize = '1.5em';

    // Create image element
    const studentImage = document.createElement('img');
    studentImage.src = imageUrl;
    studentImage.style.margin = '20px';
    studentImage.style.maxWidth = '200px';
    studentImage.style.maxHeight = '200px';
    studentImage.style.borderRadius = '8px';
    studentImage.alt = 'Student Image';

    // Create error message element
    const errorText = document.createElement('p');
    errorText.textContent = errorMessage;
    errorText.style.margin = '10px';
    errorText.style.fontSize = '1.2em';
    errorText.style.color = '#ffcccc';

    // Append elements to overlay
    overlay.appendChild(failureText);
    overlay.appendChild(nameText);
    overlay.appendChild(locationText);
    overlay.appendChild(studentImage);
    overlay.appendChild(errorText);

    // Add to document
    document.body.appendChild(overlay);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        window.screenShown = false;
        if (document.body.contains(overlay)) {
            document.body.removeChild(overlay);
        }
    }, 5000);
}

function loadDestinationOptions() {
    const destinationSelect = document.getElementById('destinationSelect');
    destinationSelect.innerHTML = ''; // Clear existing options
    
    if (window.destinationLocationJson) {
        Object.values(window.destinationLocationJson).forEach(location => {
            const option = document.createElement('option');
            option.value = location[0];
            option.textContent = location[0];
            destinationSelect.appendChild(option);
        });
    } else {
        const option = document.createElement('option');
        option.value = '';
        option.textContent = 'No destinations available';
        destinationSelect.appendChild(option);
    }
}

async function submitDestination() {
    closeDPopup();

    const destinationSelect = document.getElementById('destinationSelect');
    const selectedDestination = destinationSelect.value;

    if (!selectedDestination) {
        createAlertPopup(5000, 'warning', 'No Destination Selected', 'Please select a destination before submitting.');
        return;
    }
    
    // Get the corresponding destination ID
    let destinationId = null;
    if (window.destinationLocationJson) {
        for (const [key, value] of Object.entries(window.destinationLocationJson)) {
            if (value[0] === selectedDestination) {
                destinationId = key;
                break;
            }
        }
    }
    if (!destinationId) {
        createAlertPopup(5000, 'error', 'Invalid Destination', 'The selected destination is invalid.');
        return;
    }
    
    var studentid = await getStudentIdFromCard(window.cardid);
    if (studentid == null) {
        failBEEP.play();
        createAlertPopup(5000, null, 'Error', 'No student ID found for card');
        dlog('No student ID found for card:', window.cardid);
        return;
    }

    var floorName = await getStudentFloorName(studentid);
    if (floorName === 'error') {
        failBEEP.play();
        createAlertPopup(5000, null, 'Error', 'Failed to fetch student floor name');
        dlog('Error fetching floor name for student ID:', studentid);
        return;
    }
    
    createNewStudentPass(studentid, floorName, destinationId).then(passid => {
        if (passid !== 'error') {
            approvePassByCard(window.cardid);
        }
    });
}

// Function to show camera selection modal
function showCameraSelectionModal(cameras) {
    return new Promise((resolve) => {
        // Create modal overlay
        const overlay = document.createElement('div');
        overlay.id = 'cameraSelectionOverlay';
        overlay.style.position = 'fixed';
        overlay.style.top = '0';
        overlay.style.left = '0';
        overlay.style.width = '100%';
        overlay.style.height = '100%';
        overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
        overlay.style.display = 'flex';
        overlay.style.justifyContent = 'center';
        overlay.style.alignItems = 'center';
        overlay.style.zIndex = '10000';
        
        // Create modal content
        const modal = document.createElement('div');
        modal.style.backgroundColor = '#fff';
        modal.style.padding = '20px';
        modal.style.borderRadius = '8px';
        modal.style.maxWidth = '500px';
        modal.style.width = '80%';
        
        // Create title
        const title = document.createElement('h2');
        title.textContent = 'Select Camera';
        title.style.marginTop = '0';
        modal.appendChild(title);
        
        // Create description
        const description = document.createElement('p');
        description.textContent = 'Please select which camera to use for face recognition:';
        description.style.marginBottom = '20px';
        modal.appendChild(description);
        
        // Create select element for cameras
        const select = document.createElement('select');
        select.id = 'cameraSelect';
        select.style.width = '100%';
        select.style.padding = '10px';
        select.style.marginBottom = '20px';
        select.style.fontSize = '16px';
        
        // Add camera options
        cameras.forEach((camera, index) => {
            const option = document.createElement('option');
            option.value = camera.deviceId;
            option.textContent = camera.label || `Camera ${index + 1}`;
            select.appendChild(option);
        });
        
        modal.appendChild(select);
        
        // Create confirm button
        const confirmButton = document.createElement('button');
        confirmButton.textContent = 'Use Selected Camera';
        confirmButton.style.padding = '10px 20px';
        confirmButton.style.backgroundColor = '#4CAF50';
        confirmButton.style.color = 'white';
        confirmButton.style.border = 'none';
        confirmButton.style.borderRadius = '4px';
        confirmButton.style.cursor = 'pointer';
        confirmButton.style.fontSize = '16px';
        
        confirmButton.addEventListener('click', () => {
            const selectedCameraId = select.value;
            document.body.removeChild(overlay);
            resolve(selectedCameraId);
        });
        
        modal.appendChild(confirmButton);
        
        overlay.appendChild(modal);
        document.body.appendChild(overlay);
    });
}

// Function to get available cameras and prompt user to select one
async function setupCameraSelection() {
    try {
        // Check if media devices are available
        if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
            console.warn('Media Devices API not available');
            return null;
        }
        
        // First request camera access to get permission and device labels
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        
        // Stop all tracks to release the camera
        stream.getTracks().forEach(track => track.stop());
        
        // Get all video devices
        const devices = await navigator.mediaDevices.enumerateDevices();
        const cameras = devices.filter(device => device.kind === 'videoinput');
        
        if (cameras.length === 0) {
            console.warn('No cameras found');
            return null;
        } else if (cameras.length === 1) {
            // If only one camera, use it automatically
            return cameras[0].deviceId;
        } else {
            // If multiple cameras, show selection modal
            return await showCameraSelectionModal(cameras);
        }
    } catch (error) {
        console.error('Error setting up camera selection:', error);
        return null;
    }
}

// Function to preload face recognition models
async function preloadFaceRecognitionModels() {
    try {
        if (typeof faceapi !== 'undefined') {
            statusText.textContent = "Preloading face recognition models...";
            await faceapi.nets.tinyFaceDetector.loadFromUri('/static/face/models');
            await faceapi.nets.faceLandmark68Net.loadFromUri('/static/face/models');
            await faceapi.nets.faceRecognitionNet.loadFromUri('/static/face/models');
            window.faceApiModelsLoaded = true;
            dlog("Face recognition models preloaded successfully");
        } else {
            console.warn("faceapi.js not loaded yet, cannot preload models");
        }
    } catch (error) {
        console.error("Error preloading face recognition models:", error);
        window.faceApiModelsLoaded = false;
    }
}

document.addEventListener('DOMContentLoaded', async () => {
    enterFullscreenForDocument();
    
    // Preload face recognition models on page load
    preloadFaceRecognitionModels();
    
    // Set up camera selection first
    window.selectedCameraId = await setupCameraSelection();
    
    if (window.destinationLocationJson == null) {
        window.destinationLocationJson = await getLocationId(1);
    }
    if (window.floorLocationJson == null) {
        window.floorLocationJson = await getLocationId(2);
    }

    loadDestinationOptions();

    const locationText = document.getElementById('locationText');
    getUserLocation().then(location => {
        if (location && location !== 'error') {
            locationText.textContent = `Location: ${location}`;
        } else {
            locationText.textContent = 'Location: Unknown';
        }
    }).catch(error => {
        failBEEP.play();
        window.errorLog.push(error);
        locationText.textContent = 'Location: Error fetching location';
    });
});

async function processCardData(data) {
    if (!window.cardInAction && !window.screenShown) {
            try {
                window.cardInAction = true;
                window.cardid = data;
                dlog('Card scanned with ID:', data);
    	        let approveStudentStatus = await approveStudent(data);
                if (approveStudentStatus === 'NOPASS') {
                    if (await determineUserLocationType() === 2) {
                        openDPopup()
                    } else {
                        let studentImage = await getStudentImage(await getStudentIdFromCard(data));
                        if (studentImage === 'error') {
                            studentImage = '/static/resource/studentImagePlaceholder.png';
                        }
                        let studentName = await getStudentInfo(await getStudentIdFromCard(data));
                        if (studentName === 'error') {
                            studentName = 'Unknown';
                        } else {
                            studentName = studentName[0];
                        }

                        let studentFloorName = await getStudentFloorName(await getStudentIdFromCard(data));
                        if (studentFloorName === 'error') {
                            studentFloorName = 'Unknown';
                        }

                        failBEEP.play();

                        showFailureScreen(studentName || 'Unknown', studentFloorName || 'Unknown', studentImage || '/static/resource/studentImagePlaceholder.png', 'No pass found. Please find your duty teacher.');
                    }
                    return;
                }
            } catch (error) {
                failBEEP.play();
                window.errorLog.push(error);
                dlog('Error processing card scan:', error);
                createAlertPopup(5000, null, 'Error', 'Error processing card scan');
            } finally {
                window.cardInAction = false;
            }
        }
}

function isSEBKiosk() {
    return navigator.userAgent.includes('SEBKIOSK');
}

document.addEventListener('keydown', async function(event) {
    if (event.ctrlKey && event.key === 'e') {
        if (!isSEBKiosk()) {
            return;
        }
        event.preventDefault();
        const { value: password } = await Swal.fire({
          title: "KIOSK Quit Password",
          input: "password",
          inputLabel: "Please enter your password to quit the KIOSK.",
          inputPlaceholder: "Enter your password",
          inputAttributes: {
            autocapitalize: "off",
            autocorrect: "off"
          }
        });        
        try {
            const response = await fetch('/api/checkUserPassword', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({password: password})
            });
            const responseJson = await response.json();

        switch (responseJson.status) {
            case "error":
                failBEEP.play();
                createAlertPopup(5000, null, 'Error While Checking User Password', responseJson.errorinfo);
                return 'error';
            case "ok":
                if (responseJson.correct) {
                    try {
                        window.location.href = 'https://exabyte.org.cn/quitKIOSK'
                    } catch (error) {
                        
                    }
                } else {
                    failBEEP.play();
                    createAlertPopup(5000, null, 'Error', 'Incorrect Password');
                }
            default:
                return 'error';
        }
        } catch (error) {
            failBEEP.play();
            window.errorLog.push(error);
            dlog('Error:', error);
            createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        }
    }
});

window.cardInAction = false;
window.HIDData = '';

if (!isMobileDevice()) {
    window.CardScannerMonitor({prefix: ';', suffix: '?'}, async function(data){
        processCardData(data);
    });
} else {
    document.getElementById('cardidInput').addEventListener('input', async (event) => {
        const data = event.target.value;
        event.target.value = '';
        window.HIDData += data
        if (data == '?') {
            processCardData(window.HIDData.slice(1, -1));
            window.HIDData = ''
        }
    }
    );
    setInterval(function() {document.getElementById('cardidInput').focus()}, 3000)
}