async function confirmDialog(title, text, icon, buttonText) {
    var result = await Swal.fire({
        title: title,
        text: text,
        icon: icon,
        showCancelButton: true,
        confirmButtonColor: "#3085d6",
        cancelButtonColor: "#d33",
        confirmButtonText: buttonText
    })

    if (result.isConfirmed) {
        return true
    }

    return false
}

function getParameterByName(name) {
    return new URLSearchParams(window.location.search).get(name);
}

function createAlertPopup(closetimeout, type = null, title, body, alertid = '') {
    let alertContainer = document.getElementById('alertContainer');
    let alertElement = document.createElement('div');
    alertElement.setAttribute('id', alertid);
    alertElement.classList.add('alert');
    if (type) alertElement.classList.add(type);

    let closeButton = document.createElement('span');
    closeButton.classList.add('closebtn');
    closeButton.setAttribute('onclick', "this.parentElement.remove()");
    closeButton.innerHTML = '&times;';

    let titleText = document.createElement('strong');
    titleText.textContent = title;

    let bodyText = document.createElement('p');
    bodyText.textContent = body;

    alertElement.appendChild(closeButton);
    alertElement.appendChild(titleText);
    alertElement.appendChild(bodyText);

    alertContainer.appendChild(alertElement);

    if (closetimeout != null) {
        setTimeout(() => alertElement.remove(), closetimeout);
    }
}

async function fetchStudentImage() {
    try {
        const response = await fetch(`/api/student/getStudentImage`);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const data = await response.json();
        const imageElement = document.getElementById('studentImage');
        if (data.image) {
            imageElement.src = data.image;
        } else {
            imageElement.src = '/static/images/studentImagePlaceholder.png'; // Default image if no data
        }
    } catch (error) {
        console.error('Error fetching student image:', error);
        createAlertPopup(5000, null, 'Error', 'Failed to fetch student image');
        return null;
    }
}

async function fetchStudentName() {
    try {
        const response = await fetch(`/api/student/getStudentName`);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return await response.json();
    }
    catch (error) {
        console.error('Error fetching student info:', error);
        createAlertPopup(5000, null, 'Error', 'Failed to fetch student information');
        return null;
    }
}

async function fetchStudentPassInfo() {
    try {
        const response = await fetch(`/api/student/getStudentPassInfo`);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return await response.json();
    } catch (error) {
        console.error('Error fetching student pass info:', error);
        createAlertPopup(5000, null, 'Error', 'Failed to fetch student pass information');
        return null;
    }
}

async function loadStudentImage() {
    var studentImage = await fetchStudentImage();
    if (!studentImage) return;
    document.getElementById('studentImage').src = studentImage.src;
}

async function loadStudentName() {
    var studentName = await fetchStudentName();
    studentName = studentName.name
    if (!studentName) return;
    document.getElementById('studentName').textContent = studentName;
}

async function loadStudentPassInfo() {
    var studentPassInfo = await fetchStudentPassInfo();
    if (!studentPassInfo) return;

    const passText = document.getElementById('passInfo');
    const addPassButton = document.getElementById('addPassButton');
    const deletePassButton = document.getElementById('deletePassButton')

    passText.innerHTML = studentPassInfo.passinfo;

    addPassButton.disabled = false
    deletePassButton.disabled = true

    if (studentPassInfo.passstatus) {
        deletePassButton.disabled = false
        addPassButton.disabled = true
        if (studentPassInfo.approved) {
            deletePassButton.disabled = true
        }
    }
}

function toggleOverlay(status) {
    const overlay = document.getElementById('popupOverlay');
    if (status) {
        overlay.style.display = 'flex';
    } else {
        overlay.style.display = 'none';
    }
}

async function fetchDestinations() {
    try {
        const response = await fetch(`/api/student/getDestinations`);
        if (!response.ok) throw new Error('Network response was not ok');
        return await response.json();
    } catch (error) {
        createAlertPopup(5000, null, 'Error', 'Failed to fetch destinations');
        return null;
    }
}

async function createStudentPass(destinationName) {
    try {
        const response = await fetch(`/api/student/newPass`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ destinationname: destinationName })
        });
        const data = await response.json();
        return data;
    } catch (error) {
        createAlertPopup(5000, null, 'Error', 'Failed to create pass');
        return { status: 'error', errorinfo: error.toString() };
    }
}

async function showCreatePassPopup() {
    toggleOverlay(true);
    const select = document.getElementById('destinationSelect');
    select.innerHTML = `<option value="">Loading...</option>`;
    const data = await fetchDestinations();
    select.innerHTML = '';
    if (data && data.status === 'ok' && Array.isArray(data.destinations)) {
        select.innerHTML = '<option value="">Select Destination</option>';
        data.destinations.forEach(dest => {
            const opt = document.createElement('option');
            opt.value = dest;
            opt.textContent = dest;
            select.appendChild(opt);
        });
    } else {
        select.innerHTML = '<option value="">No destinations found</option>';
    }
}

async function deletePass() {
    try {
        const response = await fetch(`/api/student/deletePass`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        const data = await response.json();
        if (data.status == 'ok') {
            createAlertPopup(5000, 'success', 'Pass Deleted', 'Successfully deleted your most recent unused pass')
            loadStudentPassInfo();
        } else {
            createAlertPopup(5000, null, 'Error', data.errorinfo)
        }
    } catch (error) {
        createAlertPopup(5000, null, 'Error', 'Failed to create pass');
        return { status: 'error', errorinfo: error.toString() };
    }
}

document.addEventListener('contextmenu', event => {
    event.preventDefault();
});

document.addEventListener('DOMContentLoaded', async () => {
    const signoutButton = document.getElementById('signoutButton');
    const refreshButton = document.getElementById('refreshButton');
    const createPassButton = document.getElementById('addPassButton');
    const popupCloseButton = document.getElementById('overlayCloseBtn');
    const deletePassButton = document.getElementById('deletePassButton');

    signoutButton.onclick = async function () { if (await confirmDialog('Sign Out', 'Do you want to signout now?', 'warning', 'Signout')) {window.location = '/signout';} };
    refreshButton.onclick = function () { loadStudentPassInfo(); };
    createPassButton.onclick = function () { showCreatePassPopup(); };
    popupCloseButton.onclick = function () { toggleOverlay(false); }
    deletePassButton.onclick = async function () { if (await confirmDialog('Delete Pass', 'Are you sure you want to delete your pass?', 'warning', 'Delete')) {deletePass(); loadStudentPassInfo();}}

    loadStudentName();
    loadStudentImage();

    const suspensionInfo = document.getElementById('suspensionInfo')
    const actionButtons = document.getElementById('actionButtons')
    const passInfoDiv = document.getElementById('passInfoDiv')

    if (suspensionInfo.innerHTML != '') {
        actionButtons.innerHTML = ''
        passInfoDiv.innerHTML = ''
        return 0
    }

    loadStudentPassInfo();

    document.getElementById('submitNewPassButton').onclick = async function () {
        const select = document.getElementById('destinationSelect');
        const destination = select.value;
        if (!destination) {
            createAlertPopup(3000, null, 'Error', 'Please select a destination');
            return;
        }
        const result = await createStudentPass(destination);
        if (result.status === 'ok') {
            createAlertPopup(3000, 'success', 'Success', 'Pass created successfully');
            loadStudentPassInfo();
            toggleOverlay(false);
            loadStudentPassInfo();
        } else {
            createAlertPopup(5000, null, 'Error', result.errorinfo || 'Failed to create pass');
        }
    };

    setInterval(() => {loadStudentPassInfo()}, 300000)
});