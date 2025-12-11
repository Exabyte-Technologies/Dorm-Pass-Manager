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

window.debug = false
function dlog(text) {
    if (window.debug) {
        console.log(text)
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

function toDatetimeLocal(dateString) {
    const date = new Date(dateString);
    const pad = n => n < 10 ? '0' + n : n;
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
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
                createAlertPopup(5000, null, 'Error While Getting Location ID', response.errorinfo)
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
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        return 'error'
    }
}

async function searchStudents(filters) {
    try {
        const response = await fetch("/api/searchStudents", {
            method: 'POST',
            body: JSON.stringify({
                "searchFilter": filters
            }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        const responseJson = await response.json();

        switch (responseJson.status) {
            case "error":
                createAlertPopup(5000, null, 'Error While Searching For Students', response.errorinfo)
                return 'error'
                break;
            case "ok":
                return responseJson.students
                break;
            default:
                return None
        }
        
    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        return 'error'
    }
}

async function searchUsers(filters) {
    try {
        const response = await fetch("/api/searchUsers", {
            method: 'POST',
            body: JSON.stringify({
                "searchFilter": filters
            }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        const responseJson = await response.json();

        switch (responseJson.status) {
            case "error":
                createAlertPopup(5000, null, 'Error While Searching For Users', response.errorinfo)
                return 'error'
                break;
            case "ok":
                return responseJson.users
                break;
            default:
                return None
        }

    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        return 'error'
    }
}

async function searchLocations(filters) {
    try {
        const response = await fetch("/api/searchLocations", {
            method: 'POST',
            body: JSON.stringify({
                "searchFilters": filters
            }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        const responseJson = await response.json();

        switch (responseJson.status) {
            case "error":
                createAlertPopup(5000, null, 'Error While Searching For Locations', responseJson.errorinfo)
                return 'error'
                break;
            case "ok":
                return responseJson.locations
                break;
            default:
                return None
        }

    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        return 'error'
    }
}

async function addUser(name, email, role, location, password = null) {
    try {
        const response = await fetch("/api/addUser", {
            method: 'POST',
            body: JSON.stringify({
                "name": name,
                "email": email,
                "role": role,
                "location": location,
                "password": password
            }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        const responseJson = await response.json();

        switch (responseJson.status) {
            case "error":
                createAlertPopup(5000, null, 'Error While Adding User', responseJson.errorinfo)
                return 'error'
            case "ok":
                return responseJson.userid
            default:
                return None
        }
        
    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        return 'error'
    }
}

async function addStudent(studentName, studentGrade, studentFloor, studentCardid, studentImage, studentEmail) {
    try {
        const response = await fetch("/api/addStudent", {
            method: 'POST',
            body: JSON.stringify({
                "name": studentName,
                "grade": studentGrade,
                "floor": studentFloor,
                "cardid": studentCardid,
                "email": studentEmail,
                "image": studentImage
            }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        const responseJson = await response.json();

        switch (responseJson.status) {
            case "error":
                createAlertPopup(5000, null, 'Error While Adding Students', responseJson.errorinfo)
                return 'error'
            case "ok":
                return responseJson.studentid
            default:
                return None
        }
        
    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        return 'error'
    }
}

async function addLocation(locationName, locationType) {
    try {
        const response = await fetch("/api/addLocation", {
            method: 'POST',
            body: JSON.stringify({
                "name": locationName,
                "type": locationType
            }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        const responseJson = await response.json();

        switch (responseJson.status) {
            case "error":
                createAlertPopup(5000, null, 'Error Adding Location', responseJson.errorinfo)
                return 'error'
            case "ok":
                return responseJson.locationid
            default:
                return None
        }

    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        return 'error'
    }
}

async function editStudent(studentid, studentName, studentGrade, studentFloor, studentCardid, studentImage, studentEmail, studentSuspensionMessage, studentSuspensionEndDate) {
    try {
        const response = await fetch("/api/editStudent", {
            method: 'POST',
            body: JSON.stringify({
                "studentid": studentid,
                "name": studentName,
                "grade": studentGrade,
                "floor": studentFloor,
                "cardid": studentCardid,
                "email": studentEmail,
                "image": studentImage,
                "suspension": studentSuspensionMessage,
                "suspensionED": studentSuspensionEndDate
            }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        const responseJson = await response.json();

        switch (responseJson.status) {
            case "error":
                createAlertPopup(5000, null, 'Error Editing Students', responseJson.errorinfo)
                return 'error'
            case "ok":
                return 0
            default:
                return None
        }
        
    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        return 'error'
    }
}

async function editUser(userid, userName, userEmail, userRole, userLocation, userPassword) {
    try {
        const response = await fetch("/api/editUser", {
            method: 'POST',
            body: JSON.stringify({
                "userid": userid,
                "name": userName,
                "email": userEmail,
                "role": userRole,
                "location": userLocation,
                "password": userPassword
            }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        const responseJson = await response.json();

        switch (responseJson.status) {
            case "error":
                createAlertPopup(5000, null, 'Error Editing User', responseJson.errorinfo)
                return 'error'
            case "ok":
                return 0
            default:
                return None
        }

    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        return 'error'
    }
}

async function editLocation(locationid, locationName, locationType) {
    try {
        const response = await fetch("/api/editLocation", {
            method: 'POST',
            body: JSON.stringify({
                "locationid": locationid,
                "name": locationName,
                "type": locationType
            }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        const responseJson = await response.json();

        switch (responseJson.status) {
            case "error":
                createAlertPopup(5000, null, 'Error Editing Location', responseJson.errorinfo)
                return 'error'
            case "ok":
                return 0
            default:
                return None
        }

    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        return 'error'
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
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        return 'error'
    }
}

async function getLocationInfo(locationFilter) {
    try {
        const response = await fetch("/api/getLocationInfo", {
            method: 'POST',
            body: JSON.stringify({
                "filters": locationFilter
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
                return responseJson.locationinfo
                break;
            default:
                return None
        }
        
    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        return 'error'
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
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        return 'error'
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
                createAlertPopup(5000, null, 'Error While Updating User Location', responseJson.errorinfo)
                return 'error'
                break;
            case "ok":
                return 0
                break;
            default:
                return None
        }
        
    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        return 'error'
    }
}

function createAlertPopup(closetimeout, type = null, title, body, alertid = '') {
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

    let bodyText = document.createElement('p')
    bodyText.innerHTML = body

    alertElement.appendChild(closeButton)
    alertElement.appendChild(titleText)
    alertElement.appendChild(bodyText)

    alertContainer.appendChild(alertElement)

    if (closetimeout != null) {
        let closeTimout = setTimeout(function () {alertElement.remove()}, closetimeout);
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

async function setUsernameTopbar(username) {
    const usernameTopbarDispaly = document.getElementById("usernameTopbar")
    usernameTopbarDispaly.textContent = username
}

async function setLocationSelector(locationJson, role = 3) {
    const optionSlot = document.getElementById("locationSelector")
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
    optionSlot.disabled = role != 1
  }

function doOptionUpdate(optionid = null) {
    const studentsOption = document.getElementById('studentsOption');
    const usersOption = document.getElementById('usersOption');
    const locationsOption = document.getElementById('locationsOption');
    const commandOption = document.getElementById('commandOption');
    const backupOption = document.getElementById('backupOption');
    const settingsOption = document.getElementById('settingsOption');

    switch (optionid) {
        case 'studentsOption':
            studentsOption.classList.add('optionActivated')
            break;
        case 'usersOption':
            usersOption.classList.add('optionActivated')
            break;
        case 'locationsOption':
            locationsOption.classList.add('optionActivated')
            break;
        case 'commandOption':
            commandOption.classList.add('optionActivated')
            break;
        case 'backupOption':
            backupOption.classList.add('optionActivated')
            break;
        case 'settingsOption':
            settingsOption.classList.add('optionActivated')
            break;
        case 'moreset':
            studentsOption.classList.remove('optionActivated')
            usersOption.classList.remove('optionActivated')
            locationsOption.classList.remove('optionActivated')
            commandOption.classList.remove('optionActivated')
            backupOption.classList.remove('optionActivated')
            settingsOption.classList.remove('optionActivated')

            switch (window.selectedOption) {
                case 'studentsOption':
                    studentsOption.classList.add('optionActivated')
                    break;
                case 'usersOption':
                    usersOption.classList.add('optionActivated')
                    break;
                case 'locationsOption':
                    locationsOption.classList.add('optionActivated')
                    break;
                case 'commandOption':
                    commandOption.classList.add('optionActivated')
                    break;
                case 'backupOption':
                    backupOption.classList.add('optionActivated')
                    break;
                case 'settingsOption':
                    settingsOption.classList.add('optionActivated')
                    break;
            }
            break;
        case 'reset':
            studentsOption.classList.remove('optionActivated')
            usersOption.classList.remove('optionActivated')
            locationsOption.classList.remove('optionActivated')
            commandOption.classList.remove('optionActivated')
            backupOption.classList.remove('optionActivated')
            settingsOption.classList.remove('optionActivated')
        default:
            dlog('Incorrect reference')
            break;
    }
}

function setSelectedCustomOption(alloptions, selectedOption, pinnedOption) {
    for (let i = 0; i < alloptions.length; i ++) {
        document.getElementById(alloptions[i]).classList.remove('buttonSelectorActivated')
    }
    if (selectedOption != null) {
        dlog(alloptions[selectedOption])
        document.getElementById(alloptions[selectedOption]).classList.add('buttonSelectorActivated')
    }
    if (pinnedOption != null) {
        document.getElementById(pinnedOption).classList.add('buttonSelectorActivated')
    }
}

function setSelectedOption(optionid) {
    const studentsContainer = document.getElementById('studentsContainer');
    const usersContainer = document.getElementById('usersContainer');
    const locationsContainer = document.getElementById('locationsContainer');
    const commandContainer = document.getElementById('commandContainer');
    const backupContainer = document.getElementById('backupContainer');
    const settingsContainer = document.getElementById('settingsContainer');

    studentsContainer.classList.add('containerHidden')
    usersContainer.classList.add('containerHidden')
    locationsContainer.classList.add('containerHidden')
    commandContainer.classList.add('containerHidden')
    backupContainer.classList.add('containerHidden')
    settingsContainer.classList.add('containerHidden')

    window.selectedOption = optionid
    doOptionUpdate('moreset')

    switch (optionid) {
        case 'studentsOption':
            studentsContainer.classList.remove('containerHidden')
            break;
        case 'usersOption':
            usersContainer.classList.remove('containerHidden')
            break;
        case 'locationsOption':
            locationsContainer.classList.remove('containerHidden')
            break;
        case 'commandOption':
            commandContainer.classList.remove('containerHidden')
            break;
        case 'backupOption':
            backupContainer.classList.remove('containerHidden')
            break;
        case 'settingsOption':
            settingsContainer.classList.remove('containerHidden')
            break;
        default:
            dlog('Incorrect Reference')
            break;
    }
}

async function doEditStudentDatalistUpdate(name) {
    const studentEditDatalist = document.getElementById('editStudentFoundStudents')
    studentEditDatalist.innerHTML = ''
    var studentList = await searchStudents({'name': name})
    if (studentList != [] && studentList != undefined && studentList != null) {
        for (let i = 0; i < studentList.length; i ++) {
            const option = document.createElement('option')
            option.value = studentList[i][1]
            studentEditDatalist.appendChild(option)
        }
    }
}

async function doEditLocationDatalistUpdate(name) {
    const locationEditDatalist = document.getElementById('editLocationFoundLocations')
    locationEditDatalist.innerHTML = ''
    var locationList = await searchLocations({'name': name})
    if (locationList != [] && locationList != undefined && locationList != null) {
        for (let i = 0; i < locationList.length; i ++) {
            const option = document.createElement('option')
            option.value = locationList[i][1]
            locationEditDatalist.appendChild(option)
        }
    }
}

async function doEditUserDatalistUpdate(name) {
    const userEditDatalist = document.getElementById('editUserFoundUsers')
    userEditDatalist.innerHTML = ''
    var userList = await searchUsers({'name': name})
    if (userList != [] && userList != undefined && userList != null) {
        for (let i = 0; i < userList.length; i ++) {
            const option = document.createElement('option')
            option.value = userList[i][1]
            userEditDatalist.appendChild(option)
        }
    }
}

async function doEditStudentFloorDatalistUpdate(locationName) {
    const studentEditFloorDatalist = document.getElementById('editStudentFoundFloors')
    studentEditFloorDatalist.innerHTML = ''
    var locationList = await getLocationInfo({'name': locationName})
    if (locationList != [] && locationList != undefined && locationList != null) {
        for (let i = 0; i < locationList.length; i ++) {
            const option = document.createElement('option')
            option.value = locationList[i][1]
            option.locationId = locationList[i][0]
            studentEditFloorDatalist.appendChild(option)
            dlog('Added option:', option);
        }
    }
}

async function doAddStudentFloorDatalistUpdate(locationName) {
    const studentAddFloorDatalist = document.getElementById('addStudentFoundFloors')
    studentAddFloorDatalist.innerHTML = ''
    var locationList = await getLocationInfo({'name': locationName})
    if (locationList != [] && locationList != undefined && locationList != null) {
        for (let i = 0; i < locationList.length; i ++) {
            const option = document.createElement('option')
            option.value = locationList[i][1]
            option.locationId = locationList[i][0]
            studentAddFloorDatalist.appendChild(option)
            dlog('Added option:', option);
        }
    }
}

async function doAddUserLocationDatalistUpdate(locationName) {
    const userAddLocationDatalist = document.getElementById('addUserFoundLocations')
    userAddLocationDatalist.innerHTML = ''
    var locationList = await getLocationInfo({'name': locationName})
    if (locationList != [] && locationList != undefined && locationList != null) {
        for (let i = 0; i < locationList.length; i ++) {
            const option = document.createElement('option')
            option.value = locationList[i][1]
            option.locationId = locationList[i][0]
            userAddLocationDatalist.appendChild(option)
            dlog('Added option:', option);
        }
    }
}

async function doEditUserLocationDatalistUpdate(locationName) {
    const userEditLocationDatalist = document.getElementById('editUserFoundLocations')
    if (!userEditLocationDatalist) return
    userEditLocationDatalist.innerHTML = ''
    var locationList = await getLocationInfo({'name': locationName})
    if (locationList != [] && locationList != undefined && locationList != null) {
        for (let i = 0; i < locationList.length; i ++) {
            const option = document.createElement('option')
            option.value = locationList[i][1]
            option.locationId = locationList[i][0]
            userEditLocationDatalist.appendChild(option)
            dlog('Added option:', option);
        }
    }
}

function setStudentEditDisable(status) {
    document.getElementById('editStudentName').disabled = status
    document.getElementById('editStudentGrade').disabled = status
    document.getElementById('editStudentCardid').disabled = status
    document.getElementById('editStudentFloor').disabled = status
    document.getElementById('editStudentEmail').disabled = status
    document.getElementById('editStudentSuspensionMessage').disabled = status
    document.getElementById('editStudentSuspensionEndDate').disabled = status
    document.getElementById('editStudentImage').disabled = status
}

function setUserEditDisable(status) {
    document.getElementById('editUserName').disabled = status
    document.getElementById('editUserEmail').disabled = status
    document.getElementById('editUserRole').disabled = status
    document.getElementById('editUserLocation').disabled = status
    document.getElementById('editUserPassword').disabled = status
}

function setLocationEditDisable(status) {
    document.getElementById('editLocationName').disabled = status
    document.getElementById('editLocationType').disabled = status
}

async function loadStudentInfoEdit(name) {
    var studentList = await searchStudents({'strictname': name})
    if (studentList.length > 0) {
        dlog('found student, updating')
        window.studentEditId = studentList[0][0]
        var nameInfo = studentList[0][1]
        var gradeInfo = studentList[0][2]
        var locationInfo = await getLocationInfo({'id': studentList[0][4]})
        locationInfo = locationInfo[0][1]
        var cardidInfo = studentList[0][3]
        var emailInfo = studentList[0][6]
        var suspensionMessage = studentList[0][7]
        var suspensionEndDate = toDatetimeLocal(studentList[0][8])

        setStudentEditDisable(false)

        document.getElementById('editStudentName').value = nameInfo
        document.getElementById('editStudentGrade').value = gradeInfo
        document.getElementById('editStudentCardid').value = cardidInfo
        document.getElementById('editStudentEmail').value = emailInfo
        document.getElementById('editStudentSuspensionMessage').value = suspensionMessage
        document.getElementById('editStudentSuspensionEndDate').value = suspensionEndDate
        document.getElementById('editStudentFloor').value = locationInfo
    } else {
        dlog('student not found, clearing')
        document.getElementById('editStudentName').value = ''
        document.getElementById('editStudentGrade').value = ''
        document.getElementById('editStudentCardid').value = ''
        document.getElementById('editStudentEmail').value = ''
        document.getElementById('editStudentSuspensionMessage').value = ''
        document.getElementById('editStudentSuspensionEndDate').value = ''
        document.getElementById('editStudentFloor').value = ''

        setStudentEditDisable(true)
    }
}

async function loadUserInfoEdit(name) {
    var userList = await searchUsers({'strictname': name})
    if (userList.length > 0) {
        dlog('found user, updating')
        window.userEditId = userList[0][0]
        var nameInfo = userList[0][1]
        var emailInfo = userList[0][2]
        var roleInfo = userList[0][3]
        var locationInfo = await getLocationInfo({'id': userList[0][4]})
        locationInfo = locationInfo[0][1]

        setUserEditDisable(false)

        document.getElementById('editUserName').value = nameInfo
        document.getElementById('editUserEmail').value = emailInfo
        document.getElementById('editUserLocation').value = locationInfo

        switch (roleInfo) {
            case 1:
                document.getElementById('editUserRole').value = 'admin'
                break;
            case 2:
                document.getElementById('editUserRole').value = 'proctor'
                break;
            case 3:
                document.getElementById('editUserRole').value = 'approver'
                break;
            default:
                createAlertPopup(5000, null, 'Error', 'Error while getting user role')
                break;
        }
    } else {
        dlog('user not found, clearing')
        document.getElementById('editUserName').value = ''
        document.getElementById('editUserEmail').value = ''
        document.getElementById('editUserLocation').value = ''
        document.getElementById('editUserPassword').value = ''

        setUserEditDisable(true)
    }
}

async function loadLocationInfoEdit(name) {
    var locationList = await searchLocations({'strictname': name})
    if (locationList.length > 0) {
        dlog('found location, updating')
        window.locationEditId = locationList[0][0]
        var nameInfo = locationList[0][1]
        var typeInfo = locationList[0][2]

        document.getElementById('editLocationName').value = nameInfo
        if (typeInfo == 1) {
            document.getElementById('editLocationType').value = 'destination'
        } else if (typeInfo == 2) {
            document.getElementById('editLocationType').value = 'dorm'
        } else {
            createAlertPopup(5000, null, 'Error', 'Error while getting location type')
        }

        setLocationEditDisable(false)
    } else {
        dlog('location not found, clearing')
        document.getElementById('editLocationName').value = ''
        setLocationEditDisable(true)
    }
}

function convertImageFileToBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      
      reader.onload = () => resolve(reader.result);
      reader.onerror = error => reject(error);
      
      reader.readAsDataURL(file);
    });
  }

async function doUserAdd() {
    var userName = document.getElementById('addUserName').value
    var userEmail = document.getElementById('addUserEmail').value
    var userRole = document.getElementById('addUserRole').value
    var userLocation = document.getElementById('addUserLocation').value
    var userPassword = document.getElementById('addUserPassword').value

    var addResult = await addUser(userName, userEmail, userRole, userLocation, userPassword)
    if (addResult != 'error') {
        document.getElementById('addUserName').value = ''
        document.getElementById('addUserEmail').value = ''
        document.getElementById('addUserRole').value = ''
        document.getElementById('addUserLocation').value = ''
        document.getElementById('addUserPassword').value = ''

        createAlertPopup(5000, 'success', 'Success', `User ${userName} added successfully with an ID of ${addResult}`)
    } else {
        dlog('Error add user')
    }
}

async function doStudentAdd() {
    var studentImageFile = document.getElementById('addStudentImage').files[0]

    var studentName = document.getElementById('addStudentName').value
    var studentGrade = document.getElementById('addStudentGrade').value
    var studentFloor = document.getElementById('addStudentFloor').value
    var studentCardid = document.getElementById('addStudentCardid').value.replace(';', '').replace('?', '')
    var studentEmail = document.getElementById('addStudentEmail').value
    var studentImage = ''
    if (studentImageFile != undefined && studentImageFile != null) {
        await convertImageFileToBase64(studentImageFile).then(base64 => studentImage = base64);
    }

    var addResult = await addStudent(studentName, studentGrade, studentFloor, studentCardid, studentImage, studentEmail)
    if (addResult != 'error') {
        document.getElementById('addStudentName').value = ''
        document.getElementById('addStudentGrade').value = ''
        document.getElementById('addStudentFloor').value = ''
        document.getElementById('addStudentCardid').value = ''
        document.getElementById('addStudentEmail').value = ''
        document.getElementById('addStudentImage').value = ''
        createAlertPopup(5000, 'success', 'Success', `Student ${studentName} added successfully with an ID of ${addResult}`)
    } else {
        dlog('Error add student')
    }
}

async function doLocationAdd() {
    var locationName = document.getElementById('addLocationName').value
    var locationType = document.getElementById('addLocationType').value

    var addResult = await addLocation(locationName, locationType)
    if (addResult != 'error') {
        document.getElementById('addLocationName').value = ''
        createAlertPopup(5000, 'success', 'Success', `Location ${locationName} added successfully with an ID of ${addResult}`)
    } else {
        dlog('Error add location')
    }
}

async function deleteStudent(studentid) {
    try {
        const response = await fetch("/api/editStudent", {
            method: 'POST',
            body: JSON.stringify({
                "studentid": studentid,
                "delete": 'true'
            }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });
        
        const responseJson = await response.json();

        switch (responseJson.status) {
            case "error":
                createAlertPopup(5000, null, 'Error Deleting Student', responseJson.errorinfo)
                return 'error'
            case "ok":
                return studentid
            default:
                return None
        }

    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        return 'error'
    }
}

async function deleteUser(userid) {
    try {
        const response = await fetch("/api/editUser", {
            method: 'POST',
            body: JSON.stringify({
                "userid": userid,
                "delete": 'true'
            }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });
    
        const responseJson = await response.json();

        switch (responseJson.status) {
            case "error":
                createAlertPopup(5000, null, 'Error Deleting User', responseJson.errorinfo)
                return 'error'
            case "ok":
                return userid
            default:
                return None
        }

    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        return 'error'
    }
}

async function deleteLocation(locationid) {
    try {
        const response = await fetch("/api/editLocation", {
            method: 'POST',
            body: JSON.stringify({
                "locationid": locationid,
                "delete": 'true'
            }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        const responseJson = await response.json();

        switch (responseJson.status) {
            case "error":
                createAlertPopup(5000, null, 'Error Deleting Location', responseJson.errorinfo)
                return 'error'
            case "ok":
                return locationid
            default:
                return None
        }

    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        return 'error'
    }
}

async function doStudentDelete() {
    confirmDelete = await confirmDialog('Delete Student', 'Do you want delete the selected student?', 'warning', 'Delete')
    if (confirmDelete) {
        if (window.studentEditId == undefined || window.studentEditId == null) {
            createAlertPopup(5000, null, 'Error', 'No student selected for deletion')
            return 0
        }

        var deleteResult = await deleteStudent(window.studentEditId)
        if (deleteResult != 'error') {
            createAlertPopup(5000, 'success', 'Success', `Student with id of ${window.studentEditId} deleted successfully`)
        } else {
            dlog('Error delete student')
        }

        document.getElementById('editStudentChoose').value = ''
        loadStudentInfoEdit('')
    }
}

async function doUserDelete() {
    confirmDelete = await confirmDialog('Delete User', 'Do you want to delete the selected user?', 'warning', 'Delete')
    if (confirmDelete) {
        if (window.userEditId == undefined || window.userEditId == null) {
            createAlertPopup(5000, null, 'Error', 'No user selected for deletion')
            return 0
        }

        var deleteResult = await deleteUser(window.userEditId)
        if (deleteResult != 'error') {
            createAlertPopup(5000, 'success', 'Success', `User with id of ${window.userEditId} deleted successfully`)
        } else {
            dlog('Error delete user')
        }

        document.getElementById('editUserChoose').value = ''
        loadUserInfoEdit('')
    }
}

async function doLocationDelete() {
    confirmDelete = await confirmDialog('Delete Location', 'Do you want to delete the selected location?', 'warning', 'Delete')
    if (confirmDelete) {
        if (window.locationEditId == undefined || window.locationEditId == null) {
            createAlertPopup(5000, null, 'Error', 'No location selected for deletion')
            return 0
        }

        var deleteResult = await deleteLocation(window.locationEditId)
        if (deleteResult != 'error') {
            createAlertPopup(5000, 'success', 'Success', `Location with id of ${window.locationEditId} deleted successfully`)
        } else {
            dlog('Error delete location')
        }

        document.getElementById('editLocationChoose').value = ''
        loadLocationInfoEdit('')
    }
}

async function doStudentEdit() {
    var studentImageFile = document.getElementById('editStudentImage').files[0]

    var studentName = document.getElementById('editStudentName').value
    var studentGrade = document.getElementById('editStudentGrade').value
    var studentFloor = document.getElementById('editStudentFloor').value
    var studentCardid = document.getElementById('editStudentCardid').value.replace(';', '').replace('?', '')
    var studentEmail = document.getElementById('editStudentEmail').value
    var studentSuspensionMessage = document.getElementById('editStudentSuspensionMessage').value
    var studentSuspensionEndDate = document.getElementById('editStudentSuspensionEndDate').value
    var studentImage = ''
    if (studentImageFile != undefined && studentImageFile != null) {
        await convertImageFileToBase64(studentImageFile).then(base64 => studentImage = base64);
    }

    if (window.studentEditId == undefined || window.studentEditId == null) {
        createAlertPopup(5000, null, 'Error', 'No student selected for editing')
        return 0
    }

    var editResult = await editStudent(window.studentEditId, studentName, studentGrade, studentFloor, studentCardid, studentImage, studentEmail, studentSuspensionMessage, studentSuspensionEndDate)
    if (editResult != 'error') {
        createAlertPopup(5000, 'success', 'Success', `Student ${studentName} edited successfully`)
        document.getElementById('editStudentChoose').value = ''
        loadStudentInfoEdit('')
    } else {
        dlog('Error edit student')
    }
}

async function doUserEdit() {
    var userName = document.getElementById('editUserName').value
    var userEmail = document.getElementById('editUserEmail').value
    var userRole = document.getElementById('editUserRole').value
    var userLocation = document.getElementById('editUserLocation').value
    var userPassword = document.getElementById('editUserPassword').value

    if (window.userEditId == undefined || window.userEditId == null) {
        createAlertPopup(5000, null, 'Error', 'No user selected for editing')
        return 0
    }

    var editResult = await editUser(window.userEditId, userName, userEmail, userRole, userLocation, userPassword)
    if (editResult != 'error') {
        createAlertPopup(5000, 'success', 'Success', `User ${userName} edited successfully`)
        document.getElementById('editUserChoose').value = ''
        loadUserInfoEdit('')
    } else {
        dlog('Error edit user')
    }
}

async function doLocationEdit() {
    var locationName = document.getElementById('editLocationName').value
    var locationType = document.getElementById('editLocationType').value

    if (window.locationEditId == undefined || window.locationEditId == null) {
        createAlertPopup(5000, null, 'Error', 'No location selected for editing')
        return 0
    }

    var editResult = await editLocation(window.locationEditId, locationName, locationType)
    if (editResult != 'error') {
        createAlertPopup(5000, 'success', 'Success', `Location ${locationName} edited successfully`)
        document.getElementById('editLocationChoose').value = ''
        loadLocationInfoEdit('')
    } else {
        dlog('Error edit location')
    }
}

function setOptionContentVisibility(allContent, selectedContent) {
    for (let i = 0; i < allContent.length; i ++) {
        document.getElementById(allContent[i]).classList.add('containerHidden')
    }
    dlog('unh')
    dlog(allContent[selectedContent])
    document.getElementById(allContent[selectedContent]).classList.remove('containerHidden')
}

function getUrlParameter(name) {
    name = name.replace(/[\[]/, '\\[').replace(/[\]]/, '\\]');
    const regex = new RegExp('[\\?&]' + name + '=([^&#]*)');
    const results = regex.exec(location.search);
    return results === null ? '' : decodeURIComponent(results[1].replace(/\+/g, ' '));
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
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        return null
    }
}

async function getSettingsValue() {
    try {
        const response = await fetch("/api/getSettingsValue", {
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
                createAlertPopup(5000, null, 'Error While Getting Settings Value', responseJson.errorinfo)
                return 'error'
            case "ok":
                return responseJson.settings
            default:
                return null
        }
        
    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        return null
    }
}

async function generateBackup() {
    try {
        const response = await fetch("/api/generateBackup", {
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
                createAlertPopup(5000, null, 'Error While Generating Backup', responseJson.errorinfo)
                return 'error'
            case "ok":
                createAlertPopup(5000, 'success', 'Backup Generated', 'Backup has been generated successfully. You can download it by clicking "Download Latest Backup" button.')
                loadBackupList()
                return 0
            default:
                return null
        }
        
    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        return null
    }
}


/**
 * Loads all settings values from the server and populates the settings form fields.
 */
async function loadSettingsValues() {
    const settings = await getSettingsValue();
    if (!settings || settings === 'error') {
        createAlertPopup(5000, null, 'Error', 'Failed to load settings from server');
        return;
    }

    // List of all settings fields to populate
    const fields = [
        'studentWarningTimeout',
        'studentAlertTimemout',
        'studentMinimumTimeout',
        'passkeyLength',
        'querySearchLimit',
        'keepSessionDays',
        'minGrade',
        'maxGrade',
        'minNameLength',
        'maxNameLength',
        'minCardidLength',
        'maxCardidLength',
        'minEmailLength',
        'maxEmailLength',
        'smtpServer',
        'smtpEmail',
        'serverURL',
        'msauthClientId',
        'msauthAuthority',
        'msauthScope',
        'msauthRedirectURL',
        'adminEmail'
    ];

    fields.forEach(field => {
        const el = document.getElementById(field);
        if (el && settings[field] !== undefined) {
            el.value = settings[field];
        }
    });
}

/**
 * Sends a single setting value to the server to update it.
 * @param {string} settingName 
 * @param {string|number} settingValue 
 * @returns {Promise<string>} "ok" or "error"
 */
async function setSettingValue(settingName, settingValue) {
    try {
        const response = await fetch("/api/setSettingsValue", {
            method: 'POST',
            body: JSON.stringify({
                settingName: settingName,
                settingValue: settingValue
            }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        const responseJson = await response.json();
        if (responseJson.status === "ok") {
            return "ok";
        } else {
            createAlertPopup(5000, null, 'Error Saving Setting', responseJson.errorinfo || 'Unknown error');
            return "error";
        }
    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server');
        return "error";
    }
}

async function getBackupList() {
    try {
        const response = await fetch("/api/getBackupList", {
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
                createAlertPopup(5000, null, 'Error While Getting Backup List', responseJson.errorinfo)
                return 'error'
            case "ok":
                return responseJson.backupFiles
            default:
                return null
        }
        
    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        return null
    }
}

async function loadBackupList() {
    const backupListDiv = document.getElementById('backupList');
    backupListDiv.innerHTML = ''; // Clear existing content
    const backupFiles = await getBackupList();
    if (!backupFiles || backupFiles === 'error') {
        createAlertPopup(5000, null, 'Error', 'Failed to load backup files');
        return;
    }
    if (backupFiles.length === 0) {
        backupListDiv.innerHTML = '<p>No backups available.</p>';
        return;
    }
    backupFiles.forEach(file => {
        const fileDiv = document.createElement('div');
        fileDiv.className = 'backup-file';
        fileDiv.innerHTML = `
            <span>${file}</span>
            <button class="download-button" onclick="downloadBackup('${file}')">Download</button>
            <button class="load-button" onclick="loadBackup('${file}')">Load</button>
            <button class="delete-button" onclick="deleteBackup('${file}')">Delete</button>
        `;
        backupListDiv.appendChild(fileDiv);
    });
}

async function downloadBackup(filename) {
    window.open(`/downloadBackup/${encodeURIComponent(filename)}`, '_blank');
}

async function deleteBackup(filename) {
    if (await confirmDialog('Delete Backup', `Do you want to delete ${filename}? This action cannot be undone!`, 'warning', 'Delete') != true) {
        return;
    }
    try {
        const response = await fetch("/api/deleteBackup", {
            method: 'POST',
            body: JSON.stringify({ filename: filename }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        const responseJson = await response.json();

        switch (responseJson.status) {
            case "error":
                createAlertPopup(5000, null, 'Error While Deleting Backup', responseJson.errorinfo)
                return 'error'
            case "ok":
                createAlertPopup(5000, 'success', 'Backup Deleted', `Backup ${filename} has been deleted successfully.`);
                loadBackupList(); // Refresh the backup list
                return 0
            default:
                return null
        }
        
    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server');
        return 'error';
    }
}

async function loadBackup(filename) {
    if (await confirmDialog('Load Backup', `Do you want to load ${filename}? This will override all data on the exsisting database and force everyone regardless of their current state to sign out!`, 'warning', 'Load') != true) {
        return;
    }
    try {
        const response = await fetch("/api/loadBackup", {
            method: 'POST',
            body: JSON.stringify({ filename: filename }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        const responseJson = await response.json();

        switch (responseJson.status) {
            case "error":
                createAlertPopup(5000, null, 'Error While Loading Backup', responseJson.errorinfo)
                return 'error'
            case "ok":
                createAlertPopup(10000, 'success', 'Backup Loaded', `Backup ${filename} has been loaded successfully. You will be signed out now.`);
                setTimeout(() => { window.location = '/' }, 3000)
                return 0
            default:
                return null
        }
        
    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server');
        return 'error';
    }
}

async function uploadBackup() {
    const fileInput = document.getElementById('backupFileInput');
    if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
        createAlertPopup(5000, null, 'Error', 'Please select a file to upload');
        return;
    }
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/uploadBackup', {
            method: 'POST',
            body: formData
            // Do NOT set Content-Type header here!
        });

        const responseJson = await response.json();

        switch (responseJson.status) {
            case "error":
                createAlertPopup(5000, null, 'Error While Uploading Backup', responseJson.errorinfo)
                return 'error'
            case "ok":
                createAlertPopup(5000, 'success', 'Backup Uploaded', `Backup ${file.name} has been uploaded successfully.`);
                loadBackupList(); // Refresh the backup list
                return 0
            default:
                return null
        }
        
    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server');
        return 'error';
    }
}

/**
 * Collects all settings from the form and saves them one by one.
 */
async function saveAllSettings() {
    const fields = [
        'studentWarningTimeout',
        'studentAlertTimemout',
        'studentMinimumTimeout',
        'passkeyLength',
        'querySearchLimit',
        'keepSessionDays',
        'minGrade',
        'maxGrade',
        'minNameLength',
        'maxNameLength',
        'minCardidLength',
        'maxCardidLength',
        'minEmailLength',
        'maxEmailLength',
        'smtpServer',
        'smtpEmail',
        'smtpPassword',
        'serverURL',
        'msauthClientId',
        'msauthClientSecret',
        'msauthAuthority',
        'msauthScope',
        'msauthRedirectURL',
        'adminEmail'
    ];

    let allOk = true;
    for (const field of fields) {
        const el = document.getElementById(field);
        if (el) {
            const value = el.value;
            if (value !== undefined && value !== null && value !== '') {
                const result = await setSettingValue(field, value);
                if (result !== "ok") {
                    allOk = false;
                }
            }
        }
    }
    if (allOk) {
        createAlertPopup(3000, 'success', 'Settings Saved', 'All settings have been saved successfully.');
        window.scrollTo(0, 0);
    }
}

function generateAlertScopeArray() {
    const scopeAdminValue = document.getElementById('scopeAdmin').checked;
    const scopeProctorValue = document.getElementById('scopeProctor').checked;
    const scopeApproverValue = document.getElementById('scopeApprover').checked;

    const scopeArray = [];

    if (scopeAdminValue) {
        scopeArray.push('admin');
    }
    if (scopeProctorValue) {
        scopeArray.push('proctor');
    }
    if (scopeApproverValue) {
        scopeArray.push('approver');
    }
    if (scopeArray.length === 0) {
        createAlertPopup(5000, null, 'Error', 'Please select at least one scope for the alert');
        return null;
    }

    return scopeArray;
}

async function doSendMasterCommand() {
    var commandSection = window.commandSection

    commandSection = commandSection.replace('Section', '')

    switch (commandSection) {
        case 'alert':
            var alertTitle = document.getElementById('alertTitle').value
            var alertMessage = document.getElementById('alertMessage').value
            var alertDuration = document.getElementById('alertDuration').value
            var alertType = document.getElementById('alertType').value

            var alertScope = generateAlertScopeArray();
            if (alertScope === null) {
                return 0; // Error already handled in generateAlertScopeArray
            }

            if (alertTitle == '' || alertMessage == '' || alertDuration == '' || alertType == '') {
                createAlertPopup(5000, null, 'Error', 'Please fill in all fields')
                return 0
            }

            var result = await sendMasterCommand('alert', {
                'title': alertTitle,
                'body': alertMessage,
                'closetimeout': alertDuration,
                'type': alertType
            },
            alertScope);

            if (result == 'error') {
                return 0
            }
            
            createAlertPopup(5000, 'success', 'Success', `Alert "${alertTitle}" has been sent successfully to ${alertScope.join(', ')}`);

            document.getElementById('alertTitle').value = ''
            document.getElementById('alertMessage').value = ''
            document.getElementById('alertDuration').value = ''
            document.getElementById('alertType').value = 'success'
            document.getElementById('scopeAdmin').checked = false
            document.getElementById('scopeProctor').checked = false
            document.getElementById('scopeApprover').checked = false
            
            break;
        case 'reload':
            var reloadTimeout = document.getElementById('reloadTimeout').value;

            var alertScope = generateAlertScopeArray();
            if (alertScope === null) {
                return 0; // Error already handled in generateAlertScopeArray
            }

            if (reloadTimeout == '') {
                createAlertPopup(5000, null, 'Error', 'Please fill in all fields')
                return 0
            }

            var result = await sendMasterCommand('reload', {
                'reloadtimeout': reloadTimeout
            },
            alertScope);
            if (result == 'error') {
                return 0
            }
            createAlertPopup(5000, 'success', 'Success', `Reload command has been sent successfully to ${alertScope.join(', ')}`);
            document.getElementById('reloadTimeout').value = ''
            document.getElementById('scopeAdmin').checked = false
            document.getElementById('scopeProctor').checked = false
            document.getElementById('scopeApprover').checked = false
            break;
        case 'signout':
            var signoutTimeout = document.getElementById('signoutTimeout').value;

            var alertScope = generateAlertScopeArray();
            if (alertScope === null) {
                return 0; // Error already handled in generateAlertScopeArray
            }

            if (signoutTimeout == '') {
                createAlertPopup(5000, null, 'Error', 'Please fill in all fields')
                return 0
            }

            var result = await sendMasterCommand('signout', {
                'signouttimeout': signoutTimeout
            },
            alertScope);
            if (result == 'error') {
                return 0
            }
            createAlertPopup(5000, 'success', 'Success', `Signout command has been sent successfully to ${alertScope.join(', ')}`);
            document.getElementById('signoutTimeout').value = ''
            document.getElementById('scopeAdmin').checked = false
            document.getElementById('scopeProctor').checked = false
            document.getElementById('scopeApprover').checked = false
            break;
        case 'redirect':
            var redirectUrl = document.getElementById('redirectUrl').value;
            var redirectTimeout = document.getElementById('redirectTimeout').value;
            var alertScope = generateAlertScopeArray();

            if (alertScope === null) {
                return 0; // Error already handled in generateAlertScopeArray
            }

            if (redirectUrl == '' || redirectTimeout == '') {
                createAlertPopup(5000, null, 'Error', 'Please fill in all fields')
                return 0
            }

            var result = await sendMasterCommand('redirect', {
                'redirecturl': redirectUrl,
                'redirecttimeout': redirectTimeout
            },
            alertScope);
            
            if (result == 'error') {
                return 0
            }

            createAlertPopup(5000, 'success', 'Success', `Redirect command has been sent successfully to ${alertScope.join(', ')}`);

            document.getElementById('redirectUrl').value = ''
            document.getElementById('redirectTimeout').value = ''
            document.getElementById('scopeAdmin').checked = false
            document.getElementById('scopeProctor').checked = false
            document.getElementById('scopeApprover').checked = false
            break;
    
        case 'custom':
            var customCommand = document.getElementById('customCommand').value;
            var customPayload = document.getElementById('customPayload').value;

            var alertScope = generateAlertScopeArray();
            if (alertScope === null) {
                return 0; // Error already handled in generateAlertScopeArray
            }

            if (customCommand == '') {
                createAlertPopup(5000, null, 'Error', 'Please fill in all fields')
                return 0
            }

            var result = await sendMasterCommand(customCommand, customPayload,
            alertScope);
            if (result == 'error') {
                return 0
            }

            createAlertPopup(5000, 'success', 'Success', `Custom command "${customCommand}" has been sent successfully to ${alertScope.join(', ')}`);

            document.getElementById('customCommand').value = ''
            document.getElementById('scopeAdmin').checked = false
            document.getElementById('scopeProctor').checked = false
            document.getElementById('scopeApprover').checked = false
            break;

        default:
            createAlertPopup(5000, null, 'Error', 'Unknown command section')
            return 0
        }
        
}

// Attach to the settings form submit event
document.addEventListener('DOMContentLoaded', () => {
    // ...existing code...
    const settingsForm = document.getElementById('settingsForm');
    if (settingsForm) {
        settingsForm.onsubmit = async function(event) {
            event.preventDefault();
            await saveAllSettings();
        };
    }
});

document.addEventListener('DOMContentLoaded', () => {
    mainProcess();
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
    var role = userinfo['user'][4]
    window.selectedOption = null

    if (userinfo == 'error') {
        userinfo = null
        createAlertPopup(5000, null, 'Error', 'An error occured while getting user information')
        return 0
    }

    setUsernameTopbar(username)

    setLocationSelector(destinationIds, role);  
    setLocationSelector(floorIds, role); 
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

    const studentsOption = document.getElementById('studentsOption');
    const usersOption = document.getElementById('usersOption');
    const locationsOption = document.getElementById('locationsOption');
    const commandOption = document.getElementById('commandOption');
    const backupOption = document.getElementById('backupOption');
    const settingsOption = document.getElementById('settingsOption');

    studentsOption.onmouseover = function() {doOptionUpdate('studentsOption')}
    usersOption.onmouseover = function() {doOptionUpdate('usersOption')}
    locationsOption.onmouseover = function() {doOptionUpdate('locationsOption')}
    commandOption.onmouseover = function() {doOptionUpdate('commandOption')}
    backupOption.onmouseover = function() {doOptionUpdate('backupOption')}
    settingsOption.onmouseover = function() {doOptionUpdate('settingsOption')}

    studentsOption.onmouseout = function() {doOptionUpdate('moreset')}
    usersOption.onmouseout = function() {doOptionUpdate('moreset')}
    locationsOption.onmouseout = function() {doOptionUpdate('moreset')}
    commandOption.onmouseout = function() {doOptionUpdate('moreset')}
    backupOption.onmouseout = function() {doOptionUpdate('moreset')}
    settingsOption.onmouseout = function() {doOptionUpdate('moreset')}

    studentsOption.onclick = function() {setSelectedOption('studentsOption')}
    usersOption.onclick = function() {setSelectedOption('usersOption')}
    locationsOption.onclick = function() {setSelectedOption('locationsOption')}
    commandOption.onclick = function() {setSelectedOption('commandOption')}
    backupOption.onclick = function() {setSelectedOption('backupOption')}
    settingsOption.onclick = function() {setSelectedOption('settingsOption')}

    const studentAddButton = document.getElementById('addStudentSelector')
    const studentEditButton = document.getElementById('editStudentSelector')
    studentAddButton.onmouseover = function() {setSelectedCustomOption(['addStudentSelector', 'editStudentSelector'], 0, window.studentSelectorChoice)}
    studentEditButton.onmouseover = function() {setSelectedCustomOption(['addStudentSelector', 'editStudentSelector'], 1, window.studentSelectorChoice)}
    studentAddButton.onmouseout = function() {setSelectedCustomOption(['addStudentSelector', 'editStudentSelector'], null, window.studentSelectorChoice)}
    studentEditButton.onmouseout = function() {setSelectedCustomOption(['addStudentSelector', 'editStudentSelector'], null, window.studentSelectorChoice)}
    window.studentSelectorChoice = null

    document.getElementById('addStudentSelector').onclick = function(event) {
        window.studentSelectorChoice = 'addStudentSelector'
        setSelectedCustomOption(['addStudentSelector', 'editStudentSelector'], 0, window.studentSelectorChoice)
        setOptionContentVisibility(['addStudentContent', 'editStudentContent'], 0)
    }
    document.getElementById('editStudentSelector').onclick = function(event) {
        window.studentSelectorChoice = 'editStudentSelector'
        setSelectedCustomOption(['addStudentSelector', 'editStudentSelector'], 1, window.studentSelectorChoice)
        setOptionContentVisibility(['addStudentContent', 'editStudentContent'], 1)
    }

    const userAddButton = document.getElementById('addUserSelector')
    const userEditButton = document.getElementById('editUserSelector')
    userAddButton.onmouseover = function() {setSelectedCustomOption(['addUserSelector', 'editUserSelector'], 0, window.userSelectorChoice)}
    userEditButton.onmouseover = function() {setSelectedCustomOption(['addUserSelector', 'editUserSelector'], 1, window.userSelectorChoice)}
    userAddButton.onmouseout = function() {setSelectedCustomOption(['addUserSelector', 'editUserSelector'], null, window.userSelectorChoice)}
    userEditButton.onmouseout = function() {setSelectedCustomOption(['addUserSelector', 'editUserSelector'], null, window.userSelectorChoice)}
    window.userSelectorChoice = null

    document.getElementById('addUserSelector').onclick = function(event) {
        window.userSelectorChoice = 'addUserSelector'
        setSelectedCustomOption(['addUserSelector', 'editUserSelector'], 0, window.userSelectorChoice)
        setOptionContentVisibility(['addUserContent', 'editUserContent'], 0)
    }
    document.getElementById('editUserSelector').onclick = function(event) {
        window.userSelectorChoice = 'editUserSelector'
        setSelectedCustomOption(['addUserSelector', 'editUserSelector'], 1, window.userSelectorChoice)
        setOptionContentVisibility(['addUserContent', 'editUserContent'], 1)
    }

    const locationAddButton = document.getElementById('addLocationSelector')
    const locationEditButton = document.getElementById('editLocationSelector')
    locationAddButton.onmouseover = function() {setSelectedCustomOption(['addLocationSelector', 'editLocationSelector'], 0, window.locationSelectorChoice)}
    locationEditButton.onmouseover = function() {setSelectedCustomOption(['addLocationSelector', 'editLocationSelector'], 1, window.locationSelectorChoice)}
    locationAddButton.onmouseout = function() {setSelectedCustomOption(['addLocationSelector', 'editLocationSelector'], null, window.locationSelectorChoice)}
    locationEditButton.onmouseout = function() {setSelectedCustomOption(['addLocationSelector', 'editLocationSelector'], null, window.locationSelectorChoice)}
    window.locationSelectorChoice = null

    document.getElementById('addLocationSelector').onclick = function(event) {
        window.locationSelectorChoice = 'addLocationSelector'
        setSelectedCustomOption(['addLocationSelector', 'editLocationSelector'], 0, window.locationSelectorChoice)
        setOptionContentVisibility(['addLocationContent', 'editLocationContent'], 0)
    }
    document.getElementById('editLocationSelector').onclick = function(event) {
        window.locationSelectorChoice = 'editLocationSelector'
        setSelectedCustomOption(['addLocationSelector', 'editLocationSelector'], 1, window.locationSelectorChoice)
        setOptionContentVisibility(['addLocationContent', 'editLocationContent'], 1)
    }

    const editStudentDatalist = document.getElementById('editStudentChoose')
    editStudentDatalist.oninput = function(){doEditStudentDatalistUpdate(editStudentDatalist.value)}
    editStudentDatalist.addEventListener('focusout', (event) => {
        loadStudentInfoEdit(editStudentDatalist.value)
      });

    const editUserDatalist = document.getElementById('editUserChoose')
    editUserDatalist.oninput = function(){doEditUserDatalistUpdate(editUserDatalist.value)}
    editUserDatalist.addEventListener('focusout', (event) => {
        loadUserInfoEdit(editUserDatalist.value)
    });

    const editLocationDatalistDisplay = document.getElementById('editLocationChoose')
    editLocationDatalistDisplay.oninput = function(){doEditLocationDatalistUpdate(editLocationDatalistDisplay.value)}
    editLocationDatalistDisplay.addEventListener('focusout', (event) => {
        loadLocationInfoEdit(editLocationDatalistDisplay.value)
    });

    const editStudentFloorDatalist = document.getElementById('editStudentFloor')
    editStudentFloorDatalist.oninput = function(){doEditStudentFloorDatalistUpdate(editStudentFloorDatalist.value)}

    const addStudentFloorDatalist = document.getElementById('addStudentFloor')
    addStudentFloorDatalist.oninput = function(){doAddStudentFloorDatalistUpdate(addStudentFloorDatalist.value)}

    const addUserLocationDatalist = document.getElementById('addUserLocation')
    addUserLocationDatalist.oninput = function(){doAddUserLocationDatalistUpdate(addUserLocationDatalist.value)}

    const editUserLocationInput = document.getElementById('editUserLocation')
    if (editUserLocationInput) editUserLocationInput.oninput = function(){doEditUserLocationDatalistUpdate(editUserLocationInput.value)}

    const editLocationDatalist = document.getElementById('editLocationChoose')
    editLocationDatalist.oninput = function(){doEditLocationDatalistUpdate(editLocationDatalist.value)}

    const addStudentImageField = document.getElementById('addStudentImage')
    const editStudentImageField = document.getElementById('editStudentImage')

    const editStudentClearFileButton = document.getElementById('editStudentClearFileButton')

    const addLocationButtonSubmit = document.getElementById('addLocationButtonSubmit')
    addLocationButtonSubmit.onclick = function(event) {doLocationAdd()}

    addStudentImageField.onchange = function() {
        if (this.files[0].size > 1600000) {
           createAlertPopup(5000, 'warning', 'File Size Warning', 'File size too large. Please select a file smaller than 1.5MB')
           this.value = ""
        }
    }

    editStudentImageField.onchange = function() {
        if (this.files[0].size > 1600000) {
           createAlertPopup(5000, 'warning', 'File Size Warning', 'File size too large. Please select a file smaller than 1.5MB')
           this.value = ""
        }
    }

    editStudentClearFileButton.onclick = function() {
        editStudentImageField.value = ''
    }
    
    var userLocation = await getUserLocation()
    if (userLocation != 'error' && userLocation != null) {
        document.getElementById('locationSelector').value = userLocation
    }

    const editStudentDeleteButton = document.getElementById('editStudentDeleteButton')
    editStudentDeleteButton.onclick = function(event) {doStudentDelete()}

    const editUserDeleteButton = document.getElementById('editUserDeleteButton')
    editUserDeleteButton.onclick = function(event) {doUserDelete()}

    const editLocationDeleteButton = document.getElementById('editLocationDeleteButton')
    editLocationDeleteButton.onclick = function(event) {doLocationDelete()}

    const addStudentButtonSubmit = document.getElementById('addStudentButtonSubmit')
    addStudentButtonSubmit.onclick = function(event) {doStudentAdd()}

    const editStudentButtonSubmit = document.getElementById('editStudentButtonSubmit')
    editStudentButtonSubmit.onclick = function(event) {doStudentEdit()}

    const editUserButtonSubmit = document.getElementById('editUserButtonSubmit')
    editUserButtonSubmit.onclick = function(event) {doUserEdit()}

    const editLocationButtonSubmit = document.getElementById('editLocationButtonSubmit')
    editLocationButtonSubmit.onclick = function(event) {doLocationEdit()}

    const addUserButtonSubmit = document.getElementById('addUserButtonSubmit')
    addUserButtonSubmit.onclick = function(event) {doUserAdd()}

    editStudentDatalist.value = ''
    loadStudentInfoEdit('')

    const editStudentParam = getUrlParameter('editStudent');
    if (editStudentParam) {
        document.getElementById('editStudentChoose').value = editStudentParam;
        document.getElementById('studentsOption').click();
        document.getElementById('editStudentSelector').click();
        loadStudentInfoEdit(editStudentParam);
    }

    loadSettingsValues();
    loadBackupList();

    const generateBackupButton = document.getElementById('generateBackupButton');
    generateBackupButton.onclick = async function(event) { await generateBackup() };

    const commandSubmitButton = document.getElementById('commandSubmitButton');
    commandSubmitButton.onclick = async function(event) {
        event.preventDefault(); // Prevent form submission
        await doSendMasterCommand();
    };

    const getBackupListButton = document.getElementById('getBackupListButton');
    getBackupListButton.onclick = async function(event) {
        const backupList = await loadBackupList();
        if (backupList === 'error') {
            createAlertPopup(5000, null, 'Error', 'Failed to load backup list');
            return;
        }
        // Populate the backup list UI here
        // For example, you can create a dropdown or a table to display the backups
        console.log('Backup List:', backupList);
    };

    const uploadBackupButton = document.getElementById('uploadBackupButton');
    uploadBackupButton.onclick = async function(event) { await uploadBackup() }

    // Section button IDs and corresponding content IDs
    const sections = [
        { btn: 'alertSectionBtn', content: 'alertSection' },
        { btn: 'reloadSectionBtn', content: 'reloadSection' },
        { btn: 'redirectSectionBtn', content: 'redirectSection' },
        { btn: 'signoutSectionBtn', content: 'signoutSection' },
        { btn: 'customSectionBtn', content: 'customSection' }
    ];

    function showSection(sectionId) {
        window.commandSection = sectionId
        sections.forEach(({ content, btn }) => {
            const section = document.getElementById(content);
            const button = document.getElementById(btn);
            if (section) section.style.display = (content === sectionId) ? 'block' : 'none';
            if (button) button.classList.toggle('active', content === sectionId);
        });
    }

    // Attach click handlers
    sections.forEach(({ btn, content }) => {
        const button = document.getElementById(btn);
        if (button) {
            button.addEventListener('click', () => showSection(content));
        }
    });

    // Show the first section by default
    showSection(sections[0].content);
};

/**
 * Sends a master command to the server.
 * @param {string} commandType - The type of command to send.
 * @param {object} payload - The payload for the command.
 * @param {any} roles - The roles to target (if applicable).
 * @returns {Promise<object>} - The response JSON from the server.
 */
async function sendMasterCommand(commandType, payload, roles) {
    try {
        const res = await fetch('/api/sendMasterCommand', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                Accept: 'application/json'
            },
            body: JSON.stringify({ command: commandType, payload, roles })
        });
        return await res.json();
    } catch (error) {
        dlog('Error:', error);
        return { status: 'error', errorinfo: error.message };
    }
}