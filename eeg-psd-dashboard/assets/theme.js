// Theme toggle functionality
window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        toggle_theme_class: function (theme) {
            if (theme === 'dark') {
                document.body.classList.add('dark-mode');
            } else {
                document.body.classList.remove('dark-mode');
            }
            return window.dash_clientside.no_update;
        }
    }
});
