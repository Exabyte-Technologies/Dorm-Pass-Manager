window.debug = false;

function dlog(text) {
    if (window.debug) {
        console.log(text);
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
        setTimeout(() => alertElement.remove(), closetimeout);
    }
}

async function fetchStudentInfo(studentid) {
    try {
        const response = await fetch("/api/getStudentInfo", {
            method: 'POST',
            body: JSON.stringify({ "studentid": studentid }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        const responseJson = await response.json();

        if (responseJson.status === "error") {
            createAlertPopup(5000, null, 'Error Fetching Student Info', responseJson.errorinfo);
            return 'error';
        }

        const studentInfo = responseJson.studentinfo;
        console.log(studentInfo)
        window.studentName = studentInfo[0];
        document.getElementById('studentName').textContent = studentInfo[0];
        document.getElementById('studentGrade').textContent = studentInfo[1];
        document.getElementById('studentEmail').textContent = studentInfo[5];
        document.getElementById('studentFloor').textContent = await getLocationName(studentInfo[2]);
        document.getElementById('studentCardId').textContent = studentInfo[4]; // Ensure cardid is displayed
        document.getElementById('studentLocation').textContent = await getCurrentLocation(studentid);
        document.getElementById('passStatus').textContent = await getPassStatus(studentid);
        await loadStudentImage(studentid);

        return 'ok';
    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while fetching student information');
        return 'error';
    }
}

async function loadStudentImage(studentid) {
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
            createAlertPopup(5000, null, 'Error Fetching Student Image', responseJson.errorinfo);
            return 'error';
        }

        document.getElementById('studentImage').src = responseJson.studentBase64Image;
        return 'ok';
    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while fetching student image');
        return 'error';
    }
}

async function getLocationName(locationId) {
    try {
        const response = await fetch("/api/getLocationInfo", {
            method: 'POST',
            body: JSON.stringify({ "filters": { "id": locationId } }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        const responseJson = await response.json();

        if (responseJson.status === "error") {
            createAlertPopup(5000, null, 'Error Fetching Location Name', responseJson.errorinfo);
            return 'error';
        }

        return responseJson.locationinfo[0][1];
    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while fetching location name');
        return 'error';
    }
}

async function getCurrentLocation(studentid) {
    try {
        const response = await fetch("/api/getStudents", {
            method: 'POST',
            body: JSON.stringify({ "filter": { "studentid": studentid, "all": true } }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        const responseJson = await response.json();

        if (responseJson.status === "error") {
            createAlertPopup(5000, null, 'Error Fetching Current Location', responseJson.errorinfo);
            return 'Unknown';
        }

        const passes = responseJson.students;
        if (Object.keys(passes).length === 0) {
            // No active passes, student is at their dorm floor
            return document.getElementById('studentFloor').textContent
        }

        const latestPass = passes[0]; // Assuming the latest pass is the first in the list
        const [destinationId, floorId, fleavetime, darrivetime, dleavetime, farrivetime] = [
            latestPass[3], latestPass[2], latestPass[4], latestPass[5], latestPass[6], latestPass[7]
        ];

        window.passid = latestPass[0];

        if (darrivetime != null && dleavetime == null) {
            // Student is at the destination
            return await getLocationName(destinationId);
        } else if (fleavetime != null && darrivetime == null) {
            // Student is traveling to the destination
            return `Leaving Dorm - ${document.getElementById('studentFloor').textContent}`;
        } else if (dleavetime != null && farrivetime == null) {
            // Student is traveling back to the dorm
            return `Leaving Destination - ${await getLocationName(destinationId)}`;
        } else {
            // Pass is completed, student is at their dorm floor
            return document.getElementById('studentFloor').textContent;
        }
    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while fetching current location');
        return 'Unknown';
    }
}

async function getPassStatus(studentid) {
    try {
        const response = await fetch("/api/getStudents", {
            method: 'POST',
            body: JSON.stringify({ "filter": { "studentid": studentid, "all": true } }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        const responseJson = await response.json();

        if (responseJson.status === "error") {
            createAlertPopup(5000, null, 'Error Fetching Pass Status', responseJson.errorinfo);
            return 'Unknown';
        }

        const passes = responseJson.students;
        if (Object.keys(passes).length === 0) {
            // No active passes
            return "Inactive";
        }

        const latestPass = passes[0]; // Assuming the latest pass is the first in the list
        const [fleavetime, darrivetime, dleavetime, farrivetime] = [
            latestPass[4], latestPass[5], latestPass[6], latestPass[7]
        ];

        if (darrivetime != null && dleavetime == null) {
            // Student is at the destination
            return "Active";
        } else if (fleavetime != null && darrivetime == null) {
            // Student is traveling to the destination
            return "Active";
        } else if (dleavetime != null && farrivetime == null) {
            // Student is traveling back to the dorm
            return "Active";
        } else {
            // Pass is completed
            return "Inactive";
        }
    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while fetching pass status');
        return 'Unknown';
    }
}

async function getUserInfo() {
    try {
        const response = await fetch("/api/getUserInfo", {
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
                createAlertPopup(5000, null, 'Error While Getting User Info', responseJson.errorinfo)
                return 'error'
                break;
            case "ok":
                return responseJson.userinfo
                break;
            default:
                return None
        }
        
    } catch (error) {
        window.errorLog.push(error)
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        return 0
    }
}

document.addEventListener('DOMContentLoaded', async () => {
    const urlParams = new URLSearchParams(window.location.search);
    const studentid = urlParams.get('studentid');

    var userinfo = await getUserInfo()
    var role = userinfo['user'][4]

    if (role > 1) {
        document.getElementById('editStudentButton').style = "display: none;"
    }
    if (role > 2) {
        document.getElementById('createPassButton').style = "display: none;"
    }

    if (studentid) {
        await fetchStudentInfo(studentid);
    }

    document.getElementById('createPassButton').onclick = () => {
        window.location.href = `/newPassEdit?studentid=${studentid}`;
    };
    
    document.getElementById('editStudentButton').onclick = () => {
        window.parent.postMessage({ message: 'redirect', redirectUrl: `/managePanel?editStudent=${window.studentName}` }, '*');
    }

    document.getElementById('studentLocation').onclick = () => {
        window.parent.postMessage({ message: 'redirect', redirectUrl: `/passCatalog?viewPass=${window.passid}` }, '*');
    }

});
