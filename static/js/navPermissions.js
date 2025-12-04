// applyNavPermissions: disables top-bar nav buttons according to numeric role
// role mapping: 1 -> admin (all enabled), 2 -> proctor (disable manage), 3 -> approver (disable students and manage), default -> only passes enabled
(function () {
    function safeGet(id) {
        try { return document.getElementById(id) } catch (e) { return null }
    }

    window.applyNavPermissions = function(role) {
        const btnPasses = safeGet('passesNavButton')
        const btnStudents = safeGet('studentsNavButton')
        const btnManage = safeGet('manageNavButton')
        const btnSettings = safeGet('settingsNavButton')

        function enable(btn) {
            if (!btn) return
            btn.classList.remove('nav-disabled')
            btn.removeAttribute('aria-disabled')
            try { btn.disabled = false } catch(e){}
        }
        function disable(btn) {
            if (!btn) return
            btn.classList.add('nav-disabled')
            btn.setAttribute('aria-disabled', 'true')
            try { btn.disabled = true } catch(e){}
        }

        // enable all first
        [btnPasses, btnStudents, btnManage, btnSettings].forEach(enable)

        if (role == 1) {
            // admin - nothing to disable
            return
        }

        if (role == 2) {
            // proctor - disable manage
            disable(btnManage)
            return
        }

        if (role == 3) {
            // approver - disable students and manage
            disable(btnStudents)
            disable(btnManage)
            return
        }

        // default/other roles - only allow passes
        disable(btnStudents)
        disable(btnManage)
        disable(btnSettings)
    }
})();
