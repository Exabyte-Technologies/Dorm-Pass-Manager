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

window.errorLog = []
window.debug = false;

function dlog(text) {
    if (window.debug) {
        console.log(text);
    }
}

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

function intersectLists(list1, list2, list3) {
    const activeLists = [];
    for (const lst of [list1, list2, list3]) {
        if (JSON.stringify(lst) !== JSON.stringify(['nonefound'])) {
            activeLists.push(lst);
        }
    }
    
    if (activeLists.length === 0) {
        return [];
    }
    
    let result = activeLists[0];
    for (let i = 1; i < activeLists.length; i++) {
        result = result.filter(element => activeLists[i].includes(element));
    }
    
    return result;
}

async function createNewStudentPass(studentid, destinationid) {
    try {
        const response = await fetch("/api/newPass", {
            method: 'POST',
            body: JSON.stringify({
                "studentid": studentid,
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
        window.errorLog.push(error)
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server');
        return 'error';
    }
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
                createAlertPopup(5000, null, 'Error Updating Pass', responseJson.errorinfo);
                return 'error';
            case "ok":
                return responseJson;
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

async function getStudents(filters) {
    try {
        const response = await fetch("/api/getStudents", {
            method: 'POST',
            body: JSON.stringify({
                "filter": filters
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

async function searchFilters(searchFilters) {
    try {
        const response = await fetch("/api/searchFilters", {
            method: 'POST',
            body: JSON.stringify({
                "filter": searchFilters
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
            createAlertPopup(5000, null, 'Error', data.errorinfo || 'Unknown error');
            return null;
        }
    } catch (error) {
        window.errorLog.push(error)
        createAlertPopup(5000, null, 'Error', 'Failed to fetch student ID from card');
        return null;
    }
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
        var studentCreatePassButton = document.createElement('button')
        var studentEditButton = document.createElement('button')
        
        studentli.appendChild(studentNameDiv);
        studentli.appendChild(studentInfoDiv);
        studentli.appendChild(studentActionButton);
        studentli.appendChild(studentCreatePassButton);
        studentli.appendChild(studentEditButton);

        studentNameDiv.classList.add('studentNameDiv')
        studentInfoDiv.classList.add('studentInfoDiv')
        studentActionButton.classList.add('studentActionButton')
        studentCreatePassButton.classList.add('studentActionButton')
        studentEditButton.classList.add('studentActionButton')

        studentNameDiv.innerHTML = curstudent[0]

        studentInfoDiv.innerHTML = curstudent[1]
        studentActionButton.innerHTML = 'Actions'
        studentCreatePassButton.innerHTML = 'Add Pass'
        studentEditButton.innerHTML = 'Edit'

        studentActionButton.setAttribute('id', `actions-${curstudent[0]}`)
        studentCreatePassButton.setAttribute('id', `addpass-${curstudent[0]}`)
        studentEditButton.setAttribute('id', `edit-${curstudent[0]}`)
                                                                                                                                                                                                                                    
        document.getElementById(`actions-${curstudent[0]}`).onclick = function(event){
            renderStudentInfoPopup(curstudent[3]); // Pass the student ID to render the student info popup
        }

        document.getElementById(`addpass-${curstudent[0]}`).onclick = function(event){
            renderCreateNewPassPopup(curstudent[3])
        }

        document.getElementById(`edit-${curstudent[0]}`).onclick = function(event){
            window.location = `/managePanel?editStudent=${curstudent[2]}`
        }
    })
}

async function setStudents(studentList) {
    if (window.destinationLocationJson == null) {
        window.destinationLocationJson = await getLocationId(1)
    }
    if (window.floorLocationJson == null) {
        window.floorLocationJson = await getLocationId(2)
    }
    var studentsInformation = await studentList
    if (studentsInformation == 'error') {
        return 0
    }
    //window.opo = studentsInformation
    let studentsJson = {}
    for (l = 0; l < studentsInformation.length; l ++) {
        let floorName = window.floorLocationJson[studentsInformation[l][4]]
        let studentGrade = studentsInformation[l][2]
        let studentName = studentsInformation[l][1]
        let passid = studentsInformation[l][0]
        let flagged = studentsInformation[l][8]

        studentsJson[l] = [studentName, `Grade ${studentGrade} ${floorName}`, studentsInformation[l][1], passid, flagged]
    }
    setStudentIndex(studentsJson)
}

async function getLocationIdFromName(locationName) {
    try {
        const response = await fetch("/api/searchLocations", {
            method: 'POST',
            body: JSON.stringify({
                "searchFilters": {'strictname': locationName}
            }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        const responseJson = await response.json();

        switch (responseJson.status) {
            case "error":
                createAlertPopup(5000, null, 'Error Fetching Location Info', responseJson.errorinfo);
                return 'error';
            case "ok":
                return responseJson.locations[0][0];
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

async function setLocationFilter(option, locationJson) {
    var optionSlot = document.getElementById(option)
    var locations = []
    var locationJson = await locationJson
    for (i = 0; i < Object.keys(locationJson).length; i ++) {
        locations[i] = locationJson[Object.keys(locationJson)[i]]
    }
    locations.forEach(curlocation => {
        const option = document.createElement('option');
        option.text = curlocation;
        optionSlot.appendChild(option);
    });
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
                createAlertPopup(5000, null, 'Error Fetching Student Info', responseJson.errorinfo);
                return 'error';
            case "ok":
                return responseJson.studentinfo;
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
        let closeTimout = setTimeout(function () { alertElement.remove() }, closetimeout);
    }
}

async function setUsernameTopbar(username) {
    const usernameTopbarDispaly = document.getElementById("usernameTopbar")
    usernameTopbarDispaly.textContent = username
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
                return 'error';
            case "ok":
                return responseJson.locationJson;
            default:
                return null;
        }
        
    } catch (error) {
        window.errorLog.push(error)
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server');
        return 'error';
    }
}

async function setLocationSelector(locationJson, role = 3) {
    const optionSlot = document.getElementById("locationSelector");
    var locations = [];
    var locationJson = await locationJson;
    for (i = 0; i < Object.keys(locationJson).length; i++) {
        locations[i] = locationJson[Object.keys(locationJson)[i]];
    }
    locations.forEach(curlocation => {
        const option = document.createElement('option');
        option.text = curlocation;
        optionSlot.appendChild(option);
    });
    optionSlot.disabled = role != 1
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
        })

        const responseJson = await response.json();

        switch (responseJson.status) {
            case "error":
                createAlertPopup(5000, null, 'Error Updating User Location', responseJson.errorinfo);
                return 'error';
            case "ok":
                return 'ok';
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

async function doStudentSearch() {
    const floorOption = document.getElementById('filterFloor')
    const locationOption = document.getElementById('filterLocation')
    const gradeOption = document.getElementById('filterGrade')

    var filterDisplayText = ''

    if (floorOption.value != 'Dorm Floor') {
        var floorStudents = await doFloorSearch()
        filterDisplayText = floorOption.value + ', '
    } else {
        var floorStudents = ['nonefound']
    }

    if (locationOption.value != 'Pass Location') {
        var locationStudents = await doLocationSearch()
        filterDisplayText += locationOption.value + ', '
    } else {
        var locationStudents = ['nonefound']
    }

    if (gradeOption.value != 'Grade') {
        var gradeStudents = await doGradeSearch()
        filterDisplayText += 'Grade ' + gradeOption.value + ', '
    } else {
        var gradeStudents = ['nonefound']
    }
    
    if (filterDisplayText == '') {
        filterDisplayText = 'No Filters Active'
    } else {
        filterDisplayText = filterDisplayText.slice(0, -2)
    }

    setFilterDisplay(filterDisplayText)

    var allStudents = intersectLists(floorStudents, locationStudents, gradeStudents)

    setStudents(allStudents)
}

async function doFloorSearch() {
    const filterFloor = document.getElementById('filterFloor')

    floorId = await getLocationIdFromName(filterFloor.value)

    var floorIds = await searchStudents({'floorid': floorId})

    return floorIds
}

async function doLocationSearch() {
    const filterLocation = document.getElementById('filterLocation')

    locationId = await getLocationIdFromName(filterLocation.value)

    var destinationIds = await getStudents({'destination': locationId})

    var allIds = []

    for (i = 0; i < Object.keys(destinationIds).length; i ++) {
        var convertedStuInfo = await searchStudents({'studentid': destinationIds[i][1]})
        allIds[Object.keys(allIds).length + i] = convertedStuInfo[0]
    }
    
    return allIds
}

async function doNameSearch() {
    const searchTextBox = document.getElementById('filterSearchInput');
    const searchName = searchTextBox.value;

    if (searchName.trim() === '') {
        setStudentIndex({});
        return;
    }

    var foundStudents = await searchStudents({'name': searchName});

    if (foundStudents == 'error') {
        createAlertPopup(5000, null, 'Error', 'Error while searching for students');
        return;
    }

    setStudents(foundStudents);
}

async function doGradeSearch() {
    const filterGrade = document.getElementById('filterGrade');
    const grade = filterGrade.value;

    if (grade.trim() === '') {
        setStudentIndex({});
        return;
    }

    var foundStudents = await searchStudents({'grade': grade});

    if (foundStudents == 'error') {
        createAlertPopup(5000, null, 'Error', 'Error while searching for students');
        return;
    }

    return foundStudents
}

function toggleOverlay(status) {
    var overlayCloseBtn = document.getElementById('overlayCloseBtn');
    var overlay = document.getElementById('popupOverlay');
    
    if (status == true) {
        overlay.style.display = 'flex'
    } else {
        overlay.style.display = 'none'
    }
}

async function renderCreateNewPassPopup(studentid) {
    var overlayContent = document.getElementById('overlayContent')
    overlayContent.innerHTML = ''

    var createPassIframe = document.createElement('iframe')
    createPassIframe.src = `/newPassEdit?studentid=${studentid}`
    createPassIframe.style.width = '600px'
    createPassIframe.style.height = '650px'
    createPassIframe.style.border = 'none'

    overlayContent.appendChild(createPassIframe)
    toggleOverlay(true)
}

async function renderStudentInfoPopup(studentid) {
    var studentid = await studentid
    var overlayContent = document.getElementById('overlayContent');
    overlayContent.innerHTML = '';

    var studentInfoIframe = document.createElement('iframe');
    studentInfoIframe.src = `/studentInfo?studentid=${studentid}`;
    studentInfoIframe.style.width = '600px';
    studentInfoIframe.style.height = '700px';
    studentInfoIframe.style.border = 'none';

    overlayContent.appendChild(studentInfoIframe);
    toggleOverlay(true);
}

window.addEventListener('message', (event) => {
    if (event.data.message === 'createnewpassclose') {
        toggleOverlay(false);
        createAlertPopup(5000, 'success', 'Pass Created', `Pass ID: ${event.data.passId} has been created successfully.`);
    }
});

window.addEventListener('message', (event) => {
    if (event.data.message === 'redirect') {
        toggleOverlay(false);
        window.location = event.data.redirectUrl;
    }
});

function setFilterDisplay(text) {
    const filterDisplay = document.getElementById('filterDisplay');
    filterDisplay.textContent = text;
}

window.CardScannerMonitor({prefix: ';', suffix: '?'}, function(data){
	renderStudentInfoPopup(getStudentIdFromCard(data))
});

async function mainProcess() {
    const filterFloorOptions = document.getElementById('filterFloor')
    const filterLocationOptions = document.getElementById('filterLocation')

    document.getElementById('usernameTopbar').onclick = async function(event) {
        var signoutNow = await confirmDialog('Sign Out', 'Do you want to signout now?', 'warning', 'Signout')
        if (signoutNow) {
            window.location = '/signout'
        }
    }

    var userinfo = await getUserInfo()
    var username = userinfo['user'][1]
    var role = userinfo['user'][4]

    if (userinfo == 'error') {
        userinfo = null
        createAlertPopup(5000, null, 'Error', 'An error occured while getting user information')
        return 0
    }

    setUsernameTopbar(username)

    var destinationIds = await getLocationId(1);
    var floorIds = await getLocationId(2);

    setLocationFilter('filterLocation', {...destinationIds});
    setLocationFilter('filterFloor', floorIds);

    setLocationSelector(destinationIds, role)
    setLocationSelector(floorIds, role)
    // Apply role-based navigation permissions
    try {
        if (typeof applyNavPermissions === 'function') applyNavPermissions(role)
    } catch (e) {
        dlog('applyNavPermissions not available')
    }

    document.getElementById('passesNavButton').onclick = function(event) {
        event.preventDefault();
        if (typeof window.navigateWithFade === 'function') {
            window.navigateWithFade('/passCatalog');
        } else {
            window.location.href = '/passCatalog';
        }
    }
    document.getElementById('studentsNavButton').onclick = function(event) {
        event.preventDefault();
        if (typeof window.navigateWithFade === 'function') {
            window.navigateWithFade('/studentCatalog');
        } else {
            window.location.href = '/studentCatalog';
        }
    }
    document.getElementById('manageNavButton').onclick = function(event) {
        event.preventDefault();
        if (typeof window.navigateWithFade === 'function') {
            window.navigateWithFade('/managePanel');
        } else {
            window.location.href = '/managePanel';
        }
    }
    document.getElementById('settingsNavButton').onclick = function(event) {
        event.preventDefault();
        if (typeof window.navigateWithFade === 'function') {
            window.navigateWithFade('/settingsPanel');
        } else {
            window.location.href = '/settingsPanel';
        }
    }

    const locationSelector = document.getElementById('locationSelector');
    locationSelector.onchange = async (event) => {
        const updateResult = await updateUserLocation(locationSelector.value);
        if (updateResult == 'error') {
            createAlertPopup(5000, null, 'Error', 'Error while updating user location data');
            return 0;
        }
    }

    setFilterDisplay('No Filter Active')

    var userLocation = await getUserLocation()
    if (userLocation != 'error' && userLocation != null) {
        document.getElementById('locationSelector').value = userLocation
    }

    filterFloorOptions.value = 'Dorm Floor'
    filterLocationOptions.value = 'Pass Location'

    filterFloorOptions.onchange = function () {doStudentSearch()}
    filterLocationOptions.onchange = function () {doStudentSearch()}

    const searchTextBox = document.getElementById('filterSearchInput');
    searchTextBox.oninput = function () { doNameSearch() };

    const filterGradeOptions = document.getElementById('filterGrade');
    filterGradeOptions.onchange = function () { doStudentSearch() };

    overlayCloseBtn.addEventListener('click', () => {
        toggleOverlay(false)
    });

    let keysPressed = {};
    window.exportingError = false

    document.addEventListener('keydown', (event) => {
        keysPressed[event.key] = true;
        if (keysPressed['Control'] && keysPressed['e'] && keysPressed['r'] && !window.exportingError) {
            window.exportingError = true
            keysPressed = {}
            if (confirm('ERROR REPORT EXPORT\nAre you sure you want to export errors to your clipboard?')) {
                navigator.clipboard.writeText(String(window.errorLog))
            }
            window.exportingError = false
    }

    document.addEventListener('keyup', (event) => {
        delete keysPressed[event.key];
    });
});
}

document.addEventListener('DOMContentLoaded', () => {
    mainProcess()
    toggleOverlay(false)
});
