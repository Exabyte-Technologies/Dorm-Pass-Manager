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

window.onbeforeunload = null;

function renderLoader() {
    const loaderslot = document.getElementById("loaderSlot")
    loaderslot.insertAdjacentHTML("afterbegin", '<div class="loader"></div>')
}

function removeLoader() {
    const loaderslot = document.getElementById("loaderSlot")
    loaderslot.innerHTML = ''
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
                console.error(response.errorinfo)
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
        console.error('Error:', error);
        createAlertPopup(null, 'Error', 'Error while sending data to server')
        return 0
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

async function doEditLocationDatalistUpdate(name) {
    const locationEditDatalist = document.getElementById('editUserFoundLocations')
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
                console.error(responseJson.errorinfo)
                return 'error'
                break;
            case "ok":
                return responseJson.userinfo
                break;
            default:
                return None
        }
        
    } catch (error) {
        console.error('Error:', error);
        createAlertPopup(null, 'Error', 'Error while sending data to server')
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
                console.error(responseJson.errorinfo)
                return 'error'
                break;
            case "ok":
                return 0
                break;
            default:
                return None
        }
        
    } catch (error) {
        console.error('Error:', error);
        createAlertPopup(null, 'Error', 'Error while sending data to server')
        return 0
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

async function setUsernameTopbar(username) {
    const usernameTopbarDispaly = document.getElementById("usernameTopbar")
    usernameTopbarDispaly.textContent = username
}

async function loadUserInfoEdit() {
    var userList = await searchUsers({'settingsEdit': 'true'})
    if (userList.length > 0) {
        dlog('found user, updating')
        window.userEditId = userList[0]
        var nameInfo = userList[1]
        var emailInfo = userList[2]
        var roleInfo = userList[3]
        var locationInfo = await getLocationInfo({'id': userList[4]})
        locationInfo = locationInfo[0][1]

        document.getElementById('editUserName').value = nameInfo
        document.getElementById('editUserEmail').value = emailInfo
        document.getElementById('editUserLocation').value = locationInfo
    } else {
        createAlertPopup(5000, null, 'Error', 'Error while loading user information')
        return 0
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

async function editUser(userName, userEmail, userLocation, userPassword) {
    try {
        const response = await fetch("/api/editUser", {
            method: 'POST',
            body: JSON.stringify({
                "settingsEdit": 'true',
                "name": userName,
                "email": userEmail,
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
                createAlertPopup(5000, null, 'Error Saving Settings', responseJson.errorinfo)
                return 'error'
            case "ok":
                return 0
            default:
                return 'error'
        }

    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        return 'error'
    }
}

async function doUserEdit() {
    renderLoader()

    var userName = document.getElementById('editUserName').value
    var userEmail = document.getElementById('editUserEmail').value
    var userLocation = document.getElementById('editUserLocation').value
    var userPassword = document.getElementById('editUserPassword').value

    var editUserResult = await editUser(userName, userEmail, userLocation, userPassword)

    if (editUserResult == 'error') {
        dlog('Settings Save Error')
    } else {
        createAlertPopup(5000, 'success', 'Settings Saved', 'Settings has been successfully saved to the server')
    }

    removeLoader()
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
                return 'error'
        }

    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        return 'error'
    }
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

async function getNewKIOSKPin() {
    try {
        const response = await fetch("/api/generateKIOSKPin", {
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
                createAlertPopup(5000, null, 'Error While Generating KIOSK Pin', responseJson.errorinfo)
                return null
            case "ok":
                createAlertPopup(5000, 'success', 'KIOSK Pin Generated', `A new KIOSK Pin has been generated`)
                return responseJson.pin
            default:
                return null
        }
    }   catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        return null
    }
}

async function getKIOSKPin() {
    try {
        const response = await fetch("/api/getKIOSKPin", {
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
                createAlertPopup(5000, null, 'Error While Getting KIOSK Pin', responseJson.errorinfo)
                return null
            case "ok":
                return responseJson.pin
            default:
                return null
        }
    }   catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
        return null
    }
}

var pin = '●●●●●●●●'
var kioskPINShown = false

async function loadKIOSKPin() {
    pin = await getKIOSKPin()
    if (pin == null) {
        return 0
    }

    if (kioskPINShown) {
        showKIOSKPin()
    }
}

async function showKIOSKPin() {
    kioskPINShown = true
    for (i = 0; i < 8; i ++) {
        document.getElementById(`kioskPIN-${i}`).value = pin[i]
    }
}

async function hideKIOSKPin() {
    kioskPINShown = false
    for (i = 0; i < 8; i ++) {
        document.getElementById(`kioskPIN-${i}`).value = '●'
    }
}

async function getKIOSKConfigURL() {
    var kioskConfigURL = await fetch("/getKioskConfigURL")
      .then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.text()
      })
      .then((data) => {
        return data        
      })
      .catch((error) => {
        console.error("Error fetching data:", error);
      });
      return kioskConfigURL
}

async function configKIOSK() {
    var kioskConfigURL = await getKIOSKConfigURL()
    Swal.fire({
       title: 'Configure KIOSK',
       text: 'Are you configuring this computer or another computer?',
       icon: 'question',
       showDenyButton: true,
       showCancelButton: true,
       confirmButtonText: 'This Computer',
       denyButtonText: 'Another Computer',
       customClass: {
         actions: 'my-actions',
         cancelButton: 'order-1 right-gap',
         confirmButton: 'order-2',
         denyButton: 'order-3',
       },
     }).then((result) => {
       if (result.isConfirmed) {
         window.open(kioskConfigURL, '_blank');
         createAlertPopup(10000, 'success', 'KIOSK Configured', 'KIOSK has been configured if you have it installed. If you do not have the KIOSK installed, please install the KIOSK from <a href="https://sourceforge.net/projects/seb/files/latest/download">here</a>.')
       } else if (result.isDenied) {
         Swal.fire('Different Computer KIOSK Config', 'Please download SEB (using this link https://sourceforge.net/projects/seb/files/latest/download) and then open this link on any browser on the computer you want to configure: ' + kioskConfigURL, 'info')
       }
     })
}

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

    document.getElementById('editUserLocation').disabled = role != 1

    if (userinfo == 'error') {
        userinfo = null
        createAlertPopup(5000, null, 'Error', 'An error occured while getting user information')
        return 0
    }

    setUsernameTopbar(username)

    setLocationSelector(destinationIds, role);  
    setLocationSelector(floorIds, role); 
    try { if (typeof applyNavPermissions === 'function') applyNavPermissions(role) } catch (e) { dlog('applyNavPermissions not available') }
    
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

    const editLocationDatalistDisplay = document.getElementById('editUserLocation')
    editLocationDatalistDisplay.oninput = function(){doEditLocationDatalistUpdate(editLocationDatalistDisplay.value)}
    editLocationDatalistDisplay.addEventListener('focusout', (event) => {
        loadLocationInfoEdit(editLocationDatalistDisplay.value)
    });

    var userLocation = await getUserLocation()
    if (userLocation != 'error' && userLocation != null) {
        document.getElementById('locationSelector').value = userLocation
    }

    const editUserButtonSubmit = document.getElementById('editUserButtonSubmit')
    editUserButtonSubmit.onclick = function () {doUserEdit()}

    const generateNewKIOSKButton = document.getElementById('generateNewKioskPin')
    generateNewKIOSKButton.onclick = async function () {renderLoader(); await getNewKIOSKPin(); await loadKIOSKPin(); removeLoader();}

    const toggleKioskPinVisibility = document.getElementById('toggleKioskPinVisibility')
    toggleKioskPinVisibility.onclick = async function () {if (kioskPINShown) {hideKIOSKPin()} else {showKIOSKPin()}}

    const configKIOSKButton = document.getElementById('configKIOSK')
    configKIOSKButton.onclick = async function () {configKIOSK()}
};

document.addEventListener('DOMContentLoaded', () => {
    mainProcess() 
    loadKIOSKPin()
    loadUserInfoEdit()
});