let socket;
let reconnectInterval;
let isSocketInitialized = false;

function createAlertPopup(closetimeout, type = null, title, body, alertid = '') {
    let alertContainer = document.getElementById('alertContainer');

    let alertElement = document.createElement('div');
    alertElement.setAttribute('id', alertid);
    alertElement.classList.add('alert');

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

async function joinRoom() {
    if (socket && socket.connected) {
        socket.emit('join', {});
    }
  }

document.addEventListener("DOMContentLoaded", () => {
    socketConnect();
});

function socketConnect() {
    // Clear any existing reconnect interval
    if (reconnectInterval) {
        clearInterval(reconnectInterval);
    }

    if (!isSocketInitialized) {
        socket = io.connect();
        isSocketInitialized = true;

        socket.on("connect", () => {
            console.log("Socket connected successfully");
            joinRoom();
        });

        socket.on("disconnect", () => {
            console.log("Socket disconnected");
        });

        socket.on("error", (error) => {
            console.error("Socket error:", error);
        });

        socket.on("command", (cmd) => {
            console.log(cmd)
            switch (cmd.command) {
                case "signout":
                    setTimeout(function(){
                    document.cookie.split(";").forEach((c) => {
                        document.cookie = c
                            .replace(/^ +/, "")
                            .replace(/=.*/, "=;expires=" + new Date(0).toUTCString() + ";path=/");
                    });

                    if ('caches' in window) {
                        caches.keys().then(function(names) {
                            for (let name of names) caches.delete(name);
                        });
                    }

                    localStorage.clear();
                    sessionStorage.clear();

                    window.onbeforeunload = function(e) {}
                    window.location = "/signout";
                }, cmd.payload.signouttimeout * 1000);
                    break;
                case "alert":
                    createAlertPopup(cmd.payload.closetimeout * 1000, cmd.payload.type, cmd.payload.title, cmd.payload.body, cmd.payload.alertid);
                    break;
                case "reload":
                    setTimeout(function() {window.location.reload();}, cmd.payload.reloadtimeout * 1000);
                    break;
                case "redirect":
                    setTimeout(function() {window.location = cmd.payload.redirecturl;}, cmd.payload.redirecttimeout * 1000);
                    break;
                default:
                    console.warn("Unknown command received:", cmd);
            }
        });

        startConnectionCheck();
    } else {
        // Socket already exists, just try to connect
        socket.connect();
    }
}

function startConnectionCheck() {
    // Check connection status every 5 seconds
    reconnectInterval = setInterval(() => {
        if (socket && !socket.connected) {
            console.warn("Socket connection lost, attempting to reconnect...");
            socket.connect();
        }
    }, 5000);
}

