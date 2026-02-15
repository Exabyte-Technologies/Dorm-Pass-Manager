window.debug = false
function dlog(text) {
    if (window.debug) {
        console.log(text)
    }
}

async function doPasswordReset(){
    renderLoader()
    var userEmail = document.getElementById('email').value
    if (await sendPasswordReset(userEmail)) {
        createAlertPopup(5000, 'success', 'Password Reset Email Sent', 'The password reset email has been sent if the email entered is linked to a valid account. Please check your inbox.')
    } else {
        dlog('Fail Pass Reset')
    }
    removeLoader()
}

async function sendPasswordReset(email) {
    try {
        const response = await fetch("/api/passwordReset", {
            method: 'POST',
            body: JSON.stringify({
                "email": email
            }),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
        });

        const responseJson = await response.json();

        switch (responseJson.status) {
            case "error":
                createAlertPopup(5000, null, 'Error While Resetting Password', responseJson.errorinfo)
                return false
                break;
            case "ok":
                return true
                break;
            default:
                return false
        }
        
    } catch (error) {
        dlog('Error:', error);
        createAlertPopup(5000, null, 'Error', 'Error while sending data to server')
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
    bodyText.textContent = body

    alertElement.appendChild(closeButton)
    alertElement.appendChild(titleText)
    alertElement.appendChild(bodyText)

    alertContainer.appendChild(alertElement)

    if (closetimeout != null) {
        let closeTimout = setTimeout(function () {alertElement.remove()}, closetimeout);
    }
}
function renderLoader() {
    const loaderslot = document.getElementById("loaderSlot")
    loaderslot.insertAdjacentHTML("afterbegin", '<div class="loader"></div>')
}
function removeLoader() {
    const loaderslot = document.getElementById("loaderSlot")
    loaderslot.innerHTML = ''
}
document.addEventListener('DOMContentLoaded', () => {
    const passwordResetSubmitButton = document.getElementById('passwordResetSubmitButton')
    passwordResetSubmitButton.onclick = function (event) {doPasswordReset()}
})