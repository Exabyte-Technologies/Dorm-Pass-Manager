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

async function fetchPassInfo(passid) {
    try {
        const response = await fetch("/api/getPassInfo", {
            method: 'POST',
            body: JSON.stringify({ "passid": passid }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        const responseJson = await response.json();

        if (responseJson.status === "error") {
            createAlertPopup(5000, null, 'Error Fetching Pass Info', responseJson.errorinfo);
            return 'error';
        }

        const passInfo = responseJson.passinfo;
        const studentInfo = responseJson.studentinfo;
        const floorInfo = responseJson.floorinfo;
        const destinationInfo = responseJson.destinationinfo;
        const passImages = responseJson.passimages;

        window.studentId = passInfo[0]

        document.getElementById('studentName').textContent = studentInfo[0];
        document.getElementById('studentGrade').textContent = studentInfo[1];
        document.getElementById('studentFloor').textContent = floorInfo[0];
        document.getElementById('destination').textContent = destinationInfo[0];
        document.getElementById('flagged').innerHTML = passInfo[3] ? "<b style='color:red'>Yes</b>" : "No";
        document.getElementById('fleavetime').innerHTML = passInfo[4] || "N/A";
        document.getElementById('darrivetime').innerHTML = passInfo[5] || "N/A";
        document.getElementById('dleavetime').innerHTML = passInfo[6] || "N/A";
        document.getElementById('farrivetime').innerHTML = passInfo[7] || "N/A";
        document.getElementById('studentImage').src = studentInfo[3];

        document.getElementById('studentName').onclick = function () {document.getElementById('studentImage').src = studentInfo[3];};

        if (document.getElementById('fleavetime').textContent != "N/A") {
            document.getElementById('fleavetime').onclick = function () {studentImage.src = passImages[0];};
            if (passImages[0] == null || passImages[0] == '') {
                document.getElementById('fleavetime').onclick = function () {studentImage.src = '/static/resource/noImageApprove.png'};
            }
        }
        if (document.getElementById('darrivetime').textContent != "N/A") {
            document.getElementById('darrivetime').onclick = function () {studentImage.src = passImages[1];};
            if (passImages[1] == null || passImages[1] == '') {
                document.getElementById('darrivetime').onclick = function () {studentImage.src = '/static/resource/noImageApprove.png'};
            }
        }
        if (document.getElementById('dleavetime').textContent != "N/A") {
            document.getElementById('dleavetime').onclick = function () {studentImage.src = passImages[2];};
            if (passImages[2] == null || passImages[2] == '') {
                document.getElementById('dleavetime').onclick = function () {studentImage.src = '/static/resource/noImageApprove.png'};
            }
        }
        if (document.getElementById('farrivetime').textContent != "N/A") {
            document.getElementById('farrivetime').onclick = function () {studentImage.src = passImages[3];};
            if (passImages[3] == null || passImages[3] == '') {
                document.getElementById('farrivetime').onclick = function () {studentImage.src = '/static/resource/noImageApprove.png'};
            }
        }

        return 'ok';
    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while fetching pass information');
        return 'error';
    }
}

async function flagPass(passid) {
    try {
        const response = await fetch("/api/updatePass", {
            method: 'POST',
            body: JSON.stringify({ "passid": passid, "flag": true }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        const responseJson = await response.json();

        if (responseJson.status === "error") {
            createAlertPopup(5000, null, 'Error Flagging Pass', responseJson.errorinfo);
            return 'error';
        }

        createAlertPopup(5000, 'success', 'Pass Flagged', 'The pass has been successfully flagged.');
        return 'ok';
    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while flagging the pass');
        return 'error';
    }
}

async function approvePass(passid) {
    try {
        const response = await fetch("/api/updatePass", {
            method: 'POST',
            body: JSON.stringify({ "passid": passid, "approve": true }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        const responseJson = await response.json();

        if (responseJson.status === "error") {
            createAlertPopup(5000, null, 'Error Approving Pass', responseJson.errorinfo);
            return 'error';
        }

        let alertType = 'success';
        let elapsedstring = 'Elapsed Time: ';
        let elapsedTime = responseJson.elapsedtime;

        // Parse elapsed time into hours, minutes, and seconds
        if (elapsedTime && Array.isArray(elapsedTime)) {
            const [hours, minutes, seconds] = elapsedTime;
            elapsedTime = ' Elapsed Time: '
            if (hours != 0) {
                if (hours > 1) {
                    elapsedTime += `${hours} Hours `;
                } else {
                    elapsedTime += `${hours} Hour `;
                }
            }
            if (minutes != 0) {
                if (minutes > 1) {
                    elapsedTime += `${minutes} Minutes `;
                } else {
                    elapsedTime += `${minutes} Minute `;
                }
            }
            if (seconds != 0) {
                if (seconds > 1) {
                    elapsedTime += `${seconds} Seconds`;
                } else {
                    elapsedTime += `${seconds} Second`;
                }
            }
        } else {
            elapsedTime = '';
        }

        if (responseJson.elapsedtimewarning === 'min') {
            alertType = 'warning';
            elapsedstring = 'The travel time of the student might be too short.';
        } else if (responseJson.elapsedtimewarning === 'warning') {
            alertType = 'warning';
            elapsedstring = 'The student has been traveling for an extended period of time.';
        } else if (responseJson.elapsedtimewarning === 'alert') {
            alertType = '';
            elapsedstring = 'The student has been traveling for too long.';
        } else {
            alertType = 'success';
            elapsedstring = 'The student has been approved.'
        }

        createAlertPopup(5000, alertType, 'Pass Approved', `${elapsedstring}${elapsedTime}`);
        setTimeout(() => window.location.reload(), 5000);
        return 'ok';
    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while approving the pass');
        setTimeout(() => window.location.reload(), 5000);
        return 'error';
    }
}

async function deletePass(passid) {
    try {
        const response = await fetch("/api/updatePass", {
            method: 'POST',
            body: JSON.stringify({ "passid": passid, "delete": "true" }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        const responseJson = await response.json();

        if (responseJson.status === "error") {
            createAlertPopup(5000, null, 'Error Deleting Pass', responseJson.errorinfo);
            return 'error';
        }

        createAlertPopup(5000, 'success', 'Pass Deleted', 'The pass has been successfully deleted.');
        setTimeout(() => window.parent.postMessage({ message: 'closeOverlay'}, '*'), 5000);
        return 'ok';
    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while deleting the pass');
        return 'error';
    }
}

document.addEventListener('DOMContentLoaded', async () => {
    const urlParams = new URLSearchParams(window.location.search);
    const passid = urlParams.get('passid');
    const reloadPass = document.getElementById('reloadPass');

    reloadPass.onclick = () => {
        window.location.reload();
    }

    if (passid) {
        await fetchPassInfo(passid);

        document.getElementById('flagPassButton').onclick = async () => {
            await flagPass(passid);
        };

        document.getElementById('approvePassButton').onclick = async () => {
            await approvePass(passid);
        };

        document.getElementById('deletePassButton').onclick = async () => {
            await deletePass(passid);
        };

        document.getElementById('viewStudentInfoButton').onclick = () => {
            window.location.href = `/studentInfo?studentid=${window.studentId}`;
        };
    } else {
        createAlertPopup(5000, null, 'Error', 'Pass ID is missing');
    }
});
