window.debug = false;

function dlog(text) {
    if (window.debug) {
        console.log(text);
    }
}

async function getLocationId(type) {
    try {
        const response = await fetch("/api/getLocationId", {
            method: 'POST',
            body: JSON.stringify({ "type": type }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        const responseJson = await response.json();

        switch (responseJson.status) {
            case "error":
                createAlertPopup(5000, null, 'Error Fetching Locations', responseJson.errorinfo);
                return 'error';
            case "ok":
                return responseJson.locationJson;
            default:
                return 'error';
        }

    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server');
        return 'error';
    }
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

        switch (responseJson.status) {
            case "error":
                createAlertPopup(5000, null, 'Error Creating Pass', responseJson.errorinfo);
                return 'error';
            case "ok":
                return responseJson.passid;
            default:
                return 'error';
        }

    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server');
        return 'error';
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
    bodyText.textContent = body;

    alertElement.appendChild(closeButton);
    alertElement.appendChild(titleText);
    alertElement.appendChild(bodyText);

    alertContainer.appendChild(alertElement);

    if (closetimeout != null) {
        let closeTimout = setTimeout(function () { alertElement.remove() }, closetimeout);
    }
}

async function loadStudentInfo(studentid) {
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

        switch (responseJson.status) {
            case "error":
                createAlertPopup(5000, null, 'Error Fetching Student Info', responseJson.errorinfo);
                return 'error';
            case "ok":
                document.getElementById('studentName').textContent = responseJson.studentinfo[0];
                document.getElementById('studentGrade').textContent = responseJson.studentinfo[1];
                const floorId = responseJson.studentinfo[2];
                const floorName = await getLocationName(floorId);
                document.getElementById('studentFloor').textContent = floorName;
                await loadStudentImage(studentid);
                return responseJson.studentinfo;
            default:
                return 'error';
        }

    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server');
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

        switch (responseJson.status) {
            case "error":
                createAlertPopup(5000, null, 'Error Fetching Student Image', responseJson.errorinfo);
                return 'error';
            case "ok":
                document.getElementById('studentImage').src = responseJson.studentBase64Image;
                return 'ok';
            default:
                return 'error';
        }

    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server');
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

        switch (responseJson.status) {
            case "error":
                createAlertPopup(5000, null, 'Error Fetching Location Name', responseJson.errorinfo);
                return 'error';
            case "ok":
                return responseJson.locationinfo[0][1];
            default:
                return 'error';
        }

    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server');
        return 'error';
    }
}

async function loadLocations() {
    const locationJson = await getLocationId(1);
    if (locationJson !== 'error') {
        const destinationSelect = document.getElementById('destination');
        for (const key in locationJson) {
            const option = document.createElement('option');
            option.value = key;
            option.text = locationJson[key];
            destinationSelect.appendChild(option);
        }
    }
}

document.addEventListener('DOMContentLoaded', async () => {
    const urlParams = new URLSearchParams(window.location.search);
    const studentid = urlParams.get('studentid');

    if (studentid) {
        await loadStudentInfo(studentid);
    }

    await loadLocations();

    document.getElementById('createPassButton').onclick = async function () {
        const destinationid = document.getElementById('destination').value;
        const floorid = document.getElementById('studentFloor').textContent; // Assuming floorid is stored in the textContent
        if (destinationid) {
            const result = await createNewStudentPass(studentid, floorid, destinationid);
            if (result !== 'error') {
                window.parent.postMessage({ message: 'createnewpassclose', passId: result }, '*');
            }
        } else {
            createAlertPopup(5000, null, 'Error', 'Please select a destination');
        }
    };
});
