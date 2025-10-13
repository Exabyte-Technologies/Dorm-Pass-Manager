console.log(`%c  _____ _______ ____  _____  
 / ____|__   __/ __ \\|  __ \\ 
| (___    | | | |  | | |__) |
 \\___ \\   | | | |  | |  ___/ 
 ____) |  | | | |__| | |     
|_____/   |_|  \\____/|_|     
                             
DANGER ZONE
注意：开发者区域

This is a browser feature intended for developers. If someone told you to copy-paste something here to enable a feature or "hack" someone's account, it is a scam and will give them access to your account. By pasting anything here, you might be putting your account and database information at risk!
If you don't understand what you are doing, please close this window.
这是一个针对开发人员的浏览器特性。如果有人告诉你在这里复制粘贴一些东西来启用某个功能或“入侵”某人的账户，这是一个骗局，他们会获得你的账户。通过在这里粘贴任何内容，您可能会将您的帐户和数据库信息置于危险之中！
如果您不明白自己在做什么，请关闭此窗口。
`, 'color: red') 

const kioskSocket = io.connect();

async function joinKioskRooms() {
    socket.emit('join', {});
  }

const okBEEP = new Audio('/static/resource/okBEEP.mp3');
const failBEEP = new Audio('/static/resource/failBEEP.mp3');

window.errorLog = []
window.debug = false
function dlog(text) {
    if (window.debug) {
        console.log(text)
    }
}

window.showCompletedPasses = false

window.onbeforeunload = function(e) {
    return 'Are you sure you want to leave this page?  You will lose any unsaved data.';
  };

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

function calculateElapsedTime(timeStr) {
    if (timeStr == null) {
        return 'N/A'
    }

    const inputTime = new Date(timeStr);
    
    const currentTime = new Date();
    
    const diffMs = currentTime - inputTime;
    
    const minutes = Math.round(diffMs / (1000 * 60));
    
    return minutes;
}

function getLocationIdByName(locationName) {
    // Search destination locations
    if (window.destinationLocationJson) {
        for (const [id, nameArr] of Object.entries(window.destinationLocationJson)) {
            if (Array.isArray(nameArr) && nameArr[0] === locationName) {
                return id;
            }
        }
    }
    // Search floor locations
    if (window.floorLocationJson) {
        for (const [id, nameArr] of Object.entries(window.floorLocationJson)) {
            if (Array.isArray(nameArr) && nameArr[0] === locationName) {
                return id;
            }
        }
    }
    return null; // Not found
}

function triggerDisplayUpdate() {
    document.getElementById(window.lastfilterchoice).click()
}

function renderLoader() {
    const loaderslot = document.getElementById("loaderSlot")
    loaderslot.insertAdjacentHTML("afterbegin", '<div class="loader"></div>')
}

function removeLoader() {
    const loaderslot = document.getElementById("loaderSlot")
    loaderslot.innerHTML = ''
}

function getLocationType(location) {
    if (!window.destinationLocationJson || !window.floorLocationJson) {
        return null;
    }

    for (const [id, name] of Object.entries(window.destinationLocationJson)) {
        if (location == id || location == name) {
            return 'destination';
        }
    }
    for (const [id, name] of Object.entries(window.floorLocationJson)) {
        if (location == id || location == name) {
            return 'floor';
        }
    }
    return null;
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
        window.errorLog.push(error)
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        return 0
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

async function generateKioskToken() {
    try {
        const response = await fetch("/api/generateKioskToken", {
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
                createAlertPopup(5000, null, 'Error While Starting KIOSK', responseJson.errorinfo)
                return 'error'
                break;
            case "ok":
                return responseJson.kioskToken
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

async function openBrowserKioskWindow() {
    var kioskConfirmResult = await confirmDialog('Launching KIOSK With Escalated Privileges', 'Launching a KIOSK from this page will give the KIOSK escalated privileges which could be exploited if an attacker gains physical access to this computer. INSTEAD, you should use another device and set the other device up using the KIOSK setup guide located in the settings panel. You could continue but this computer should be watched at all time to prevent misuse if you do. Do you want to continue?', 'warning', 'Launch')
    if (!kioskConfirmResult) {
        return
    }
  const url = '/kiosk'; // Replace with your desired URL
  const windowName = 'DPM KIOSK';
  const windowFeatures = 'popup,fullscreen';

  const newWindow = window.open(url, windowName, windowFeatures);

  if (!newWindow) {
    alert('Popup blocked! Please allow popups for this site.');
  }
}

async function startKioskMode() {
    /*
    var kioskToken = await generateKioskToken()
    if (kioskToken == 'error') {
        createAlertPopup(5000, null, 'Error', 'Error while generating kiosk token')
        return 0
    }
    window.onbeforeunload = function(e) {}
    window.dpmkiosklaunched = true
    location.replace(`dpmkiosk://${kioskToken}`)
    createAlertPopup(15000, type = 'success', 'DPM Kiosk Launched', `DPM Kiosk has been launched if you have it installed. If you do not have DPM Kiosk installed, please navigate to <a target="_blank" href="https://google.com">here</a> and download DPM Kiosk.`)
    window.onbeforeunload = function(e) { return 'Are you sure you want to leave this page?  You will lose any unsaved data.';}
    */
    openBrowserKioskWindow()
}

async function searchStudents(searchFilters) {
    try {
        const response = await fetch("/api/searchStudents", {
            method: 'POST',
            body: JSON.stringify({
                "searchFilter": searchFilters
            }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        const responseJson = await response.json();

        switch (responseJson.status) {
            case "error":
                createAlertPopup(5000, null, 'Error Fetching Students', responseJson.errorinfo);
                return 'error';
            case "ok":
                return responseJson.students;
            default:
                return 'error';
        }

    } catch (error) {
        window.errorLog.push(error)
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server');
        return 'error';
    }
}

async function getStudents(filters, allfilters = null) {
    var searchScopeOption = document.getElementById('searchOptionSlot')
    var searchScope = searchScopeOption.value
    var dateScopeOption = document.getElementById('passSearchDateNoFilter')
    var dateScopeSelect = document.getElementById('passSearchDate')
    var dateScope = ''
    var searchparams = {}
    if (!dateScopeOption.checked) {
        dateScope = dateScopeSelect.value
    }
    if (allfilters == null) {
        searchparams = {
                "filter": filters,
                "searchScope": searchScope,
                "showCompletedPass": window.showCompletedPasses,
                "dateScope": dateScope
            }
    } else {
        searchparams = allfilters
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
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        return 0
    }
}

async function addStudent(studentName, studentGrade, studentFloor, studentCardid) {
    try {
        const response = await fetch("/api/addStudent", {
            method: 'POST',
            body: JSON.stringify({
                "name": studentName,
                "grade": studentGrade,
                "floor": studentFloor,
                "cardid": studentCardid
            }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        const responseJson = await response.json();

        switch (responseJson.status) {
            case "error":
                createAlertPopup(5000, null, 'Error Adding Students', responseJson.errorinfo)
                return 'error'
            case "ok":
                return responseJson.studentid
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
        window.errorLog.push(error)
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        return 0
    }
}

async function updateUserLocation(locationName) {
    try {
        const response = await fetch("/api/updateUserLocation", {
            method: 'POST',
            body: JSON.stringify({
                "location": locationName
            }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        const responseJson = await response.json();

        switch (responseJson.status) {
            case "error":
                createAlertPopup(5000, null, 'Error While Updating Location', responseJson.errorinfo)
                return 'error'
                break;
            case "ok":
                return 0
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

function createAlertPopup(closetimeout, type = null, title, body, alertid = '', imageUrl = '', image2Url = '') {
    try {
        let alertContainer = document.getElementById('alertContainer')

        let alertElement = document.createElement('div');
        alertElement.setAttribute('id', alertid)
        alertElement.classList.add('alert')
        dlog(type)
        if (type != null && type != undefined && type != '') {
            alertElement.classList.add(type)
        }

        let closeButton = document.createElement('span');
        closeButton.classList.add('closebtn')
        closeButton.setAttribute('onclick', "this.parentElement.remove()")
        closeButton.innerHTML = '&times;'

        let titleText = document.createElement('strong')
        titleText.textContent = title

        // If imageUrl is provided, add the image to the alert
        let imageElem = null;
        if (imageUrl != '') {
            imageElem = document.createElement('img');
            imageElem.src = imageUrl;
            imageElem.style.height = '90px';
            imageElem.style.borderRadius = '8px';
            imageElem.style.display = 'inline-block';
            imageElem.style.margin = '10px auto';
        }

        let image2Elem = null;
        if (image2Url != '') {
            image2Elem = document.createElement('img');
            image2Elem.src = image2Url;
            image2Elem.style.height = '90px';
            image2Elem.style.borderRadius = '8px';
            image2Elem.style.display = 'inline-block';
            image2Elem.style.margin = '10px auto';
        }

        let bodyText = document.createElement('p')
        bodyText.innerHTML = body

        alertElement.appendChild(closeButton)
        alertElement.appendChild(titleText)

        if (imageElem != null || image2Elem != null) {
            let imageContainer = document.createElement('div');
            imageContainer.style.display = 'flex';
            if (imageElem != '' && imageElem != null) {
                imageContainer.appendChild(imageElem);
            } 
            if (image2Elem != '' && image2Elem != null) {
                imageContainer.appendChild(image2Elem);
            }
            alertElement.appendChild(imageContainer);
        }

        alertElement.appendChild(bodyText)

        alertContainer.appendChild(alertElement)

        if (closetimeout != null) {
            setTimeout(function () { alertElement.remove() }, closetimeout);
        }
    } catch (error) {
        console.log(error)
    }
}

function createDestinationChooserPopup(studentName, studentInfo, studentImage, destinationButtonJson) {
    let params = `scrollbars=no,resizable=no,status=no,location=no,toolbar=no,menubar=no,width=0,height=0,left=-1000,top=-1000`
    open(`/studentDestinationChooser?studentName=${studentName}&studentInfo=${studentInfo}&studentImage=${studentImage}&destinationButtonJson=${destinationButtonJson}`, "Choose Student Destination", params)
}

function createStudentInfoDisplayPopup() {
    let params = `scrollbars=no,resizable=no,status=no,location=no,toolbar=no,menubar=no,width=0,height=0,left=-1000,top=-1000`
    open("/studentInfoDisplay", "Student Status Display", params)
}

async function updateStudentPass(params) {
    try {
        const response = await fetch("/api/updatePass", {
            method: 'POST',
            body: JSON.stringify(params),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        const responseJson = await response.json();

        switch (responseJson.status) {
            case "error":
                createAlertPopup(5000, null, 'Error While Updating Pass', responseJson.errorinfo)
                return 'error'
            case "ok":
                window.exsistingAlert[`${params['passid'].toString()}alert`] = false
                window.exsistingAlert[`${params['passid'].toString()}warning`] = false
                try {
                    document.getElementById(`${params['passid'].toString()}alert`).remove()
                } catch (error) {
        window.errorLog.push(error)
                    dlog('No alert found to dismiss')
                }
                dlog(responseJson)
                return responseJson
            default:
                createAlertPopup(5000, null, 'Error', 'Server returned unreadable data')
                return 0
        }
        
    } catch (error) {
        window.errorLog.push(error)
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        return 0
    }
}

async function setLocationSelector(locationJson) {
    const optionSlot = document.getElementById("locationSelector")
    const searchOptionSlot = document.getElementById("searchOptionSlot")
    var locations = []
    var locationJson = await locationJson
    for (i = 0; i < Object.keys(locationJson).length; i ++) {
        locations[i] = locationJson[Object.keys(locationJson)[i]]
    }
    locations.forEach(curlocation => {
      var option = document.createElement('option');
      option.text = curlocation;
      optionSlot.appendChild(option);
    });

    locations.forEach(curlocation => {
      var option = document.createElement('option');
      option.text = curlocation;
      searchOptionSlot.appendChild(option);
    });
  }

async function setStudentIndex(studentsJson) {
    const studentListSlot = document.getElementById("studentList")

    studentListSlot.innerHTML = ""

    var students = []
    var studentsJson = await studentsJson
    for (i = 0; i < Object.keys(studentsJson).length; i ++) {
        students[i] = studentsJson[i]
    }
    students.forEach(curstudent => {
        var studentli = document.createElement('li')

        studentListSlot.appendChild(studentli)
        studentli.classList.add('list-item')
        studentli.setAttribute('id', `stu-${curstudent[2]}`)

        var studentNameDiv = document.createElement('div')
        var studentInfoDiv = document.createElement('div')
        var studentActionButton = document.createElement('button')
        var studentApproveButton = document.createElement('button')
        var studentFlagButton = document.createElement('button')

        var passid = curstudent[3]
        var flagged = curstudent[4]
        
        studentli.appendChild(studentNameDiv);
        studentli.appendChild(studentInfoDiv);
        studentli.appendChild(studentActionButton);
        studentli.appendChild(studentApproveButton);
        studentli.appendChild(studentFlagButton);

        studentNameDiv.classList.add('studentNameDiv')
        studentInfoDiv.classList.add('studentInfoDiv')
        studentActionButton.classList.add('studentActionButton')
        studentApproveButton.classList.add('studentActionButton')
        studentFlagButton.classList.add('studentActionButton')

        if (flagged == 1) {
            studentNameDiv.innerHTML = `🔴 ${curstudent[0]}`
        } else {
            studentNameDiv.innerHTML = `🟢 ${curstudent[0]}`
        }

        studentInfoDiv.innerHTML = curstudent[1]
        studentActionButton.innerHTML = 'Actions'
        studentApproveButton.innerHTML = 'Approve'
        studentFlagButton.innerHTML = 'Flag'

        studentActionButton.setAttribute('id', `actions-${curstudent[0]}-${passid}`)
        studentApproveButton.setAttribute('id', `approve-${curstudent[0]}-${passid}-${curstudent[5]}-${curstudent[6]}-${curstudent[7]}-${curstudent[8]}`)
        studentFlagButton.setAttribute('id', `flag-${curstudent[0]}-${passid}-${curstudent[5]}-${curstudent[6]}-${curstudent[7]}`)

        document.getElementById(`actions-${curstudent[0]}-${passid}`).onclick = function(event){
            renderPassInfoPopup(passid); // Open passInfo overlay
        }
        document.getElementById(`approve-${curstudent[0]}-${passid}-${curstudent[5]}-${curstudent[6]}-${curstudent[7]}-${curstudent[8]}`).onclick = async function(event){
            var userLocation = await getUserLocation()
            var locWarningStr = ''
            if (((curstudent[8] == 1) && (userLocation != curstudent[5])) || ((curstudent[8] == 2) && (userLocation != curstudent[6])) || ((curstudent[8] == 3) && (userLocation != curstudent[6])) || ((curstudent[8] == 4) && (userLocation != curstudent[5]))) {
                locWarningStr = ` | ⚠️ WARNING: ${curstudent[0]} is not supposed to be at your location`
            }
            var confirmApprove = await confirmDialog(`Approve ${curstudent[0]} ?`, `${curstudent[5]} ↔️ ${curstudent[6]} | 🕓 ${calculateElapsedTime(curstudent[7])} Min${locWarningStr}`, 'question', 'Approve')
            if (confirmApprove) {
                var updateResult = await updateStudentPass({'passid': passid,'approve': true})
                var alertType = ''
                // --- Fetch student image ---
                let studentImageUrl = null;
                try {
                    const imgResp = await fetch("/api/getStudentImage", {
                        method: 'POST',
                        body: JSON.stringify({ studentid: curstudent[2] }),
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
                } catch (e) {
                    // Ignore image errors
                }
                // --- End fetch student image ---
            
                if (updateResult != 'error') {
                    let elapsedstring = '';
                    let elapsedTime = '';
                    if (updateResult.elapsedtime != null || updateResult.elapsedtime != undefined) {
                        dlog(updateResult.elapsedtimewarning)
                        if (updateResult.elapsedtimewarning == 'min') {
                            alertType = 'warning'
                            elapsedstring = 'The travel time of the student might be too short.';
                        } else if (updateResult.elapsedtimewarning == 'warning') {
                            alertType = 'warning'
                            elapsedstring = 'The student has been traveling for an extended period of time.';
                        } else if (updateResult.elapsedtimewarning == 'alert') {
                            alertType = ''
                            elapsedstring = 'The student has been traveling for too long.';
                        } else {
                            elapsedstring = 'Elapsed Time:'
                            alertType = 'success'
                        }
                        var updateResultET = updateResult.elapsedtime
                        let elapsedTimeStr = '';
                        if (updateResultET && Array.isArray(updateResultET)) {
                            if (updateResultET[0]) elapsedTimeStr += `${updateResultET[0]} Hour${updateResultET[0] > 1 ? 's' : ''} `;
                            if (updateResultET[1]) elapsedTimeStr += `${updateResultET[1]} Minute${updateResultET[1] > 1 ? 's' : ''} `;
                            if (updateResultET[2]) elapsedTimeStr += `${updateResultET[2]} Second${updateResultET[2] > 1 ? 's' : ''}`;
                        }
                        elapsedTime = elapsedTimeStr.trim();
                    } else {
                        elapsedstring = ''
                        alertType = 'success'
                    }
                    createAlertPopup(
                        5000,
                        alertType,
                        'Approve Success',
                        `<b>${curstudent[0]}</b> has been approved. ${elapsedstring}${elapsedTime ? '<br>Elapsed Time: ' + elapsedTime : ''}`,
                        '',
                        studentImageUrl
                    );
                } else {
                    return 0
                }
                setTimeout(function () {triggerDisplayUpdate()}, 1500)
            }
        }
        document.getElementById(`flag-${curstudent[0]}-${passid}-${curstudent[5]}-${curstudent[6]}-${curstudent[7]}`).onclick = async function(event){
            var confirmApprove = await confirmDialog(`Flag ${curstudent[0]} ?`, `${curstudent[5]} ↔️ ${curstudent[6]} | 🕓 ${calculateElapsedTime(curstudent[7])} Min`, 'warning', 'Flag')
            if (confirmApprove) {
                var updateResult = updateStudentPass({'passid': passid,'flag': true})
                if (updateResult != 'error') {
                    createAlertPopup(5000, type = 'success', 'Flag Success', `${curstudent[0]} has been flagged`)
                } else {
                    return 0
                }
                triggerDisplayUpdate()
            }
        }
    })
}

async function setStudents(filters) {
    renderLoader()
    updateStudentStatusChart()
    if (window.destinationLocationJson == null) {
        window.destinationLocationJson = await getLocationId(1)
    }
    if (window.floorLocationJson == null) {
        window.floorLocationJson = await getLocationId(2)
    }
    var studentsInformation = await getStudents(filters)
    if (studentsInformation == 'error') {
        return 0
    }
    window.opo = studentsInformation
    let studentsJson = {}
    for (l = 0; l < studentsInformation.length; l ++) {
        let studentInfo  = studentsInformation[l][studentsInformation[l].length - 1]
        let floorName = window.floorLocationJson[studentInfo[2]]
        let destinationName = window.destinationLocationJson[studentsInformation[l][3]]
        let studentGrade = studentInfo[1]
        let studentName = studentInfo[0]
        let passid = studentsInformation[l][0]
        let flagged = studentsInformation[l][8]
        let passStatus = ''

        let timeoutType = studentsInformation[l][studentsInformation[l].length - 2]
        if ((timeoutType == 'alert') && window.exsistingAlert[`${passid.toString()}alert`] != true) {
            createAlertPopup(null, null, 'Student Timeout Warning', `${studentName} from ${floorName} has been inactive for too long. Please take necessary actions!`, alertid = `${passid.toString()}alert`)
            window.exsistingAlert[`${passid.toString()}alert`] = true
        } else if ((timeoutType == 'warning') && window.exsistingAlert[`${passid.toString()}warning`] != true) {
            createAlertPopup(15000, 'warning', 'Student Timeout Warning', `${studentName} from ${floorName} has been inactive for a extended period of time.`, alertid = `${passid.toString()}warning`)
            window.exsistingAlert[`${passid.toString()}warning`] = true
        }

        let lastApproveTime = null;

        for (curtime of [studentsInformation[l][7], studentsInformation[l][6], studentsInformation[l][5], studentsInformation[l][4]]) {
            if (curtime != null) {
                lastApproveTime = curtime;
                break;
            }
        }

        var studentState = 0;

        for (var i = 0; i < 4; i ++) {
            if (studentsInformation[l][i + 4] == null) {
                studentState = i + 1
                break
            }
        }

        if (studentsInformation[l][4] == null) {
            passStatus = '◽️🏢◽️'
        } else if (studentsInformation[l][5] == null) {
            passStatus = '🏢➡️⛳️'
        } else if (studentsInformation[l][6] == null) {
            passStatus = '◽️⛳️◽️'
        } else if (studentsInformation[l][7] == null) {
            passStatus = '⛳️➡️🏢'
        } else {
            passStatus = '◽️🏢✅'
        }

        studentsJson[l] = [studentName, `Grade ${studentGrade} ${floorName} - ${passStatus}`, studentsInformation[l][1], passid, flagged, floorName, destinationName, lastApproveTime, studentState]
    }
    setStudentIndex(studentsJson)
    removeLoader()
}

async function setUsernameTopbar(username) {
    const usernameTopbarDispaly = document.getElementById("usernameTopbar")
    usernameTopbarDispaly.textContent = username
}

function setFilterDisplay(text) {
    const filterDisplay = document.getElementById('filterDisplay')
    filterDisplay.textContent = text
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
                createAlertPopup(5000, null, 'Error While Getting User Location', responseJson.errorinfo)
                return 'error'
            case "ok":
                return responseJson.location
            default:
                return null
        }
        
    } catch (error) {
        window.errorLog.push(error)
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        return null
    }
}

async function approvePassByCard(cardid) {
    try {
        const response = await fetch("/api/approvePassByCard", {
            method: 'POST',
            body: JSON.stringify({ cardid }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        if (!response.ok) throw new Error('Network error');
        const responseJson = await response.json();

        var passid = responseJson.passid

        // Fetch student image if available
        let studentImageUrl = null;
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
            elapsedstring = 'The travel time of the student might be too short.';
        } else if (responseJson.elapsedtimewarning === 'warning') {
            alertType = 'warning';
            elapsedstring = 'The student has been traveling for an extended period of time.';
        } else if (responseJson.elapsedtimewarning === 'alert') {
            alertType = '';
            elapsedstring = 'The student has been traveling for too long.';
        }

        if (responseJson.status === "ok") {
            var rdn = Math.floor(Math.random() * 100000) + 1;

            createAlertPopup(
                5000,
                alertType,
                'Pass Approved',
                `${studentName ? `<b>${studentName}</b><br>` : ''}${elapsedstring}${elapsedTime ? '<br>Elapsed Time: ' + elapsedTime : ''}<br><br><button class="studentActionButton" id="info-${passid}-${rdn}">Show Info</button>`,
                '',
                studentImageUrl
            );

            document.getElementById(`info-${passid}-${rdn}`).onclick = function(event){
                    renderPassInfoPopup(passid);
                    triggerDisplayUpdate()
            }

            triggerDisplayUpdate()
        } else {
            var rdn = Math.floor(Math.random() * 100000) + 1;

            var flaggedButtonHTML = `<br><br><button class="studentActionButton" id="flag-${passid}-${rdn}">Flag</button><button class="studentActionButton" id="info-${passid}-${rdn}">Show Info</button>`

            if (passid == undefined) {
                flaggedButtonHTML = ''
            }

            createAlertPopup(
                10000,
                null,
                'Approval Failed',
                `<b>${studentName}</b><br>` + responseJson.errorinfo + flaggedButtonHTML,
                '',
                studentImageUrl
            );

            if (passid != undefined) {
                triggerDisplayUpdate()
                document.getElementById(`flag-${passid}-${rdn}`).onclick = async function(event){
                    // --- Fetch info for dialog ---
                    let floorName = '';
                    let destName = '';
                    let travelTimeStr = '';
                
                    // Fetch student info (for floor)
                    let studentInfoResp = await fetch("/api/getStudentInfo", {
                        method: 'POST',
                        body: JSON.stringify({ studentid: responseJson.studentid }),
                        headers: {
                            Accept: 'application/json',
                            'Content-Type': 'application/json',
                        },
                    });
                    if (studentInfoResp.ok) {
                        let studentInfoJson = await studentInfoResp.json();
                        if (studentInfoJson.status === "ok" && Array.isArray(studentInfoJson.studentinfo)) {
                            // If floor is an index, convert to name
                            let floorIdx = studentInfoJson.studentinfo[2];
                            if (window.floorLocationJson && window.floorLocationJson[floorIdx]) {
                                floorName = window.floorLocationJson[floorIdx];
                            } else {
                                floorName = floorIdx;
                            }
                        }
                    }
                
                    // Fetch pass info (for destination and travel time)
                    let passInfoResp = await fetch("/api/getPassInfo", {
                        method: 'POST',
                        body: JSON.stringify({
                             passid: passid,
                             appendApproverName: false
                            }),
                        headers: {
                            Accept: 'application/json',
                            'Content-Type': 'application/json',
                        },
                    });
                    if (passInfoResp.ok) {
                        let passInfoJson = await passInfoResp.json();
                        if (passInfoJson.status === "ok" && passInfoJson.passinfo) {
                            let passinfo = passInfoJson.passinfo;
                            // Adjust indices as needed for your backend
                            let floorIdx = passinfo[1];
                            let destIdx = passinfo[2];
                            if (window.floorLocationJson && window.floorLocationJson[floorIdx]) {
                                floorName = window.floorLocationJson[floorIdx];
                            }
                            if (window.destinationLocationJson && window.destinationLocationJson[destIdx]) {
                                destName = window.destinationLocationJson[destIdx];
                            }
                            // Get approve time (or last approve time)
                            let approveTime = passinfo[4] || passinfo[5] || passinfo[6] || passinfo[7];
                            if (approveTime) {
                                let travelTimeMinutes = calculateElapsedTime(approveTime);
                                travelTimeStr = `${travelTimeMinutes} Min`;
                            }
                        }
                    }
                
                    // Compose dialog text
                    let dialogText = `${floorName || ''} ↔️ ${destName || ''}`;
                    if (travelTimeStr) {
                        dialogText += ` | 🕓 ${travelTimeStr}`;
                    }
                
                    var confirmApprove = await confirmDialog(
                        `Flag ${studentName} ?`,
                        dialogText,
                        'warning',
                        'Flag'
                    );
                    if (confirmApprove) {
                        var updateResult = await updateStudentPass({'passid': passid,'flag': true});
                        if (updateResult != 'error') {
                            createAlertPopup(5000, 'success', 'Flag Success', `${studentName} has been flagged`);
                        } else {
                            createAlertPopup(5000, 'error', 'Flag Failed', `Failed to flag ${studentName}`);
                        }
                        triggerDisplayUpdate();
                    }
                }
                // ...existing code...
                document.getElementById(`info-${passid}-${rdn}`).onclick = function(event){
                    renderPassInfoPopup(passid);
                    triggerDisplayUpdate()
                }
            }
        }
    } catch (error) {
        window.errorLog.push(error)
        createAlertPopup(5000, null, 'Error', 'Error while approving pass');
    }
}

async function doResetPassFilter() {
    var searchOptionSlot = document.getElementById('searchOptionSlot')
    var passSearchDateNoFilter = document.getElementById('passSearchDateNoFilter')
    var passSearchDate = document.getElementById('passSearchDate')
    var showCompletedPass = document.getElementById('showCompletedPass')

    searchOptionSlot.value = 'Use Current Location'
    passSearchDateNoFilter.checked = true
    passSearchDate.value = ''
    showCompletedPass.checked = false

    triggerDisplayUpdate()
}

window.CardScannerMonitor({prefix: ';', suffix: '?'}, function(data){
	approvePassByCard(data)
});

async function mainProcess() {
    document.getElementById('usernameTopbar').onclick = async function(event) {
        var signoutNow = await confirmDialog('Sign Out', 'Do you want to signout now?', 'warning', 'Signout')
        if (signoutNow) {
            window.location = '/signout'
        }
    }

    var destinationIds = await getLocationId(1)
    var floorIds = await getLocationId(2)
    var userinfo = await getUserInfo()
    var username = userinfo['user'][1]
    window.lastfilterchoice = 'filterButtonRelated'
    window.exsistingAlert = {}

    const urlParams = new URLSearchParams(window.location.search);

    const firstLanding = urlParams.get('firstLanding')
    const rejectionmessage = urlParams.get('reject')
    const viewPassId = urlParams.get('viewPass')

    if (viewPassId != null && viewPassId != undefined && viewPassId != '') {
        renderPassInfoPopup(viewPassId)
    }

    if (rejectionmessage != null && rejectionmessage != undefined && rejectionmessage != '') {
        createAlertPopup(5000, null, 'Rejected From Page', rejectionmessage)
    }

    if (userinfo == 'error') {
        userinfo = null
        createAlertPopup(5000, null, 'Error', 'An error occured while getting user information')
        return 0
    }

    setUsernameTopbar(username)

    setLocationSelector(destinationIds);  
    setLocationSelector(floorIds); 

    var option = document.createElement('option');
    option.text = 'Use Current Location';
    document.getElementById("searchOptionSlot").appendChild(option);
    option.selected = true;
    
    document.getElementById('filterButtonAll').onclick = function(event) {
        window.lastfilterchoice = 'filterButtonAll'
        setStudents({'all': true})
        setFilterDisplay('All')
    }
    document.getElementById('filterButtonRelated').onclick = function(event) {
        window.lastfilterchoice = 'filterButtonRelated'
        setStudents({})
        setFilterDisplay('Related')
    }
    document.getElementById('filterButtonArriving').onclick = function(event) {
        window.lastfilterchoice = 'filterButtonArriving'
        setStudents({'status': 1})
        setFilterDisplay('Arriving')
    }
    document.getElementById('filterButtonLeaving').onclick = function(event) {
        window.lastfilterchoice = 'filterButtonLeaving'
        setStudents({'status': 3})
        setFilterDisplay('Leaving')
    }
    document.getElementById('filterButtonFlagged').onclick = function(event) {
        window.lastfilterchoice = 'filterButtonFlagged'
        setStudents({'flag': true})
        setFilterDisplay('Flagged')
    }
    document.getElementById('filterButtonPresent').onclick = function(event) {
        window.lastfilterchoice = 'filterButtonPresent'
        setStudents({'status': 2})
        setFilterDisplay('Present')
    }

    document.getElementById('passesNavButton').onclick = function(event) {
        window.location.href = '/passCatalog'
    }
    document.getElementById('studentsNavButton').onclick = function(event) {
        window.location.href = '/studentCatalog'
    }
    document.getElementById('manageNavButton').onclick = function(event) {
        window.location.href = '/managePanel'
    }
    document.getElementById('settingsNavButton').onclick = function(event) {
        window.location.href = '/settingsPanel'
    }

    const filterSearchInput = document.getElementById("filterSearchInput")
    filterSearchInput.addEventListener("input", function(event) {
        if (filterSearchInput.value != '') {
            window.searchActivated = true
            setStudents({'search':filterSearchInput.value})
            setFilterDisplay(`Search - ${filterSearchInput.value}`)
        } else {
            window.searchActivated = false
            triggerDisplayUpdate()
        }
        }
    )

    const locationSelector = window.document.getElementById('locationSelector')

    locationSelector.onchange = (event) => {
        updateUserLocation(locationSelector.value)
        if (updateUserLocation == 'error') {
            createAlertPopup(5000, null, 'Error', 'Error while updating user location data')
            return 0
        }
        if (window.searchActivated == false) {
            setTimeout(() => {triggerDisplayUpdate()}, 1000);
        }
    }

    var userLocation = await getUserLocation()
    if (userLocation != 'error' && userLocation != null) {
        document.getElementById('locationSelector').value = userLocation
    }

    if (firstLanding == 'true') {
        createAlertPopup(5000, type = 'success', 'Welcome', 'Welcome to the DPM Kiosk Management Panel')
    }

    overlayCloseBtn.addEventListener('click', () => {
        toggleOverlay(false)
    });

    window.addEventListener('message', (event) => {
        if (event.data.message === 'closeOverlay') {
            toggleOverlay(false);
        }
    });

    const showCompletedPassOption = document.getElementById('showCompletedPass')
    showCompletedPassOption.onchange = (event) => {
        window.showCompletedPasses = showCompletedPassOption.checked
        triggerDisplayUpdate()
    }

    const searchOptionSlot = document.getElementById('searchOptionSlot')

    searchOptionSlot.onchange = (event) => {triggerDisplayUpdate()}

    const passSearchDateNoFilter = document.getElementById('passSearchDateNoFilter')
    const passSearchDate = document.getElementById('passSearchDate')
    passSearchDateNoFilter.onchange = (event) => {triggerDisplayUpdate()}
    passSearchDate.onchange = (event) => {triggerDisplayUpdate()}

    const resetPassFilter = document.getElementById('resetPassFilter')
    resetPassFilter.onclick = (event) => {doResetPassFilter()}

    let keysPressed = {};
    window.exportingError = false

    document.addEventListener('keydown', async (event) => {
        keysPressed[event.key] = true;
        if (keysPressed['Control'] && keysPressed['e'] && keysPressed['r'] && !window.exportingError) {
            window.exportingError = true
            keysPressed = {}
            if (await confirmDialog('Confirm Error Report Export', 'Are you sure you want to export all errors to your clipboard?', 'warning', 'Export')) {
                navigator.clipboard.writeText(String(window.errorLog))
            }
            window.exportingError = false
    }

    document.addEventListener('keyup', (event) => {
        delete keysPressed[event.key];
    });
});

    triggerDisplayUpdate()

    window.searchActivated = false
}

function toggleOverlay(status) {
    const overlay = document.getElementById('popupOverlay');
    if (status) {
        overlay.style.display = 'flex';
    } else {
        overlay.style.display = 'none';
    }
}

async function renderPassInfoPopup(passid) {
    const overlayContent = document.getElementById('overlayContent');
    overlayContent.innerHTML = '';

    const passInfoIframe = document.createElement('iframe');
    passInfoIframe.src = `/passInfo?passid=${passid}`;
    passInfoIframe.style.width = '600px';
    passInfoIframe.style.height = '675px';
    passInfoIframe.style.border = 'none';

    overlayContent.appendChild(passInfoIframe);
    toggleOverlay(true);
}

window.addEventListener('message', (event) => {
    if (event.data.message === 'redirect') {
        toggleOverlay(false);
        window.location = event.data.redirectUrl;
    }
});

function updateDisplay() {
    if (window.searchActivated == false) {
        triggerDisplayUpdate()
    }
    updateStudentStatusChart()
}

async function updateStudentStatusChart() {
    window.destinationLocationJson = await getLocationId(1)
    window.floorLocationJson = await getLocationId(2)

    var searchScopeOption = document.getElementById('searchOptionSlot')
    var userLocationId = searchScopeOption.value
    if (userLocationId == 'Use Current Location') {
        userLocationId = getLocationIdByName(await getUserLocation())
    } else {
        userLocationId = getLocationIdByName(userLocationId)
    }
    
    if (getLocationType(userLocationId) == 'floor') {
        var totalStudents = (await searchStudents({floorid: userLocationId})).length
    } else {
        var totalStudents = (await getStudents({destination: userLocationId})).length
    }

    var notLeftStudents = (await getStudents(0, {"filter": {"status": 2}, "searchScope": searchScopeOption.value, "showCompletedPass": false, "dateScope": "" })).length
    var relatedPassStudents = (await getStudents(0, {"filter": {}, "searchScope": searchScopeOption.value, "showCompletedPass": false, "dateScope": "" })).length
    var arrivingStudents = (await getStudents(0, {"filter": {"status": 1},"searchScope": searchScopeOption.value,"showCompletedPass": false,"dateScope": ""})).length
    var leavingStudents = (await getStudents(0, {"filter": {"status": 3},"searchScope": searchScopeOption.value,"showCompletedPass": false,"dateScope": ""})).length
    var awayStudents = relatedPassStudents - notLeftStudents - arrivingStudents - leavingStudents
    var presentStudents = totalStudents - relatedPassStudents + notLeftStudents

    window.studentStatusChart.data.datasets[0].data[0] = presentStudents
    window.studentStatusChart.data.datasets[0].data[1] = awayStudents
    window.studentStatusChart.data.datasets[0].data[2] = leavingStudents
    window.studentStatusChart.data.datasets[0].data[3] = arrivingStudents

    window.studentStatusChart.update()
}

async function kioskSocketActionSetup() {
    socket.on("connect", () => {
        console.log("Socket connected successfully");
    });
    socket.on("disconnect", () => {
        console.log("Socket disconnected");
    });
    socket.on("error", (error) => {
        console.error("Socket error:", error);
    });

    socket.on("kioskUpdate", (cmd) => {
        var now = new Date();
        var currentTimeString = now.toLocaleTimeString();
        switch (cmd.status) {
            case "ok":
                var studentFloorName = window.floorLocationJson[cmd.studentFloorId][0]
                createAlertPopup(3000, 'success', 'KIOSK Approve Success', `<b>${cmd.studentName} - ${studentFloorName}</b><br>${currentTimeString}`, '', cmd.studentImage, cmd.studentScanImage);
                break;
            case "error":
                var studentFloorName = window.floorLocationJson[cmd.studentFloorId][0]
                createAlertPopup(300000, 'error', 'KIOSK Approve Error', `<b>${cmd.studentName} - ${studentFloorName}</b><br>${cmd.errorinfo}<br>${currentTimeString}`, '', cmd.studentImage, cmd.studentScanImage);
                failBEEP.play();
                break;
            default:
                console.warn("Unknown command received:", cmd);
        }
    });
    joinRoom();
}

document.addEventListener('DOMContentLoaded', () => {
    window.studentStatusChart = new Chart("studentStatusChart", {
        type: "pie",
        data: {
          labels: ["Present", "Away", "Leaving", "Arriving"],
          datasets: [{
            backgroundColor: ["#00aba9", "#b91d47", "#2b5797", "#e8c3b9",],
            data: [0, 0, 0, 0]
          }]
        },
        options: {
            plugins: {
          title: {
            font: {
                        size: 24
                    },
            display: true,
            text: "Attendance Status"
          },
          legend: {
                labels: {
                    font: {
                        size: 18
                    }
                }
            }
        }}
    });

    mainProcess()
    toggleOverlay(false)
    updateDisplay()   
    const overlayCloseBtn = document.getElementById('overlayCloseBtn');
    overlayCloseBtn.addEventListener('click', () => {
        toggleOverlay(false);
    });

    const viewPassButton = document.getElementById('viewPassButton');
    if (viewPassButton) {
        viewPassButton.onclick = () => {
            const passid = viewPassButton.getAttribute('data-passid');
            renderPassInfoPopup(passid);
        };
    }

    kioskSocketActionSetup()
    joinKioskRooms()

    setInterval(updateDisplay, 20000)
});

