document.addEventListener('DOMContentLoaded', function () {
  const resetButtons = document.querySelectorAll('button[type="reset"]');

  resetButtons.forEach(function (button) {
    button.addEventListener('click', function () {
      setTimeout(function () {
        const form = button.closest('form');
        if (form) {
          form.querySelectorAll('select, input').forEach(function (field) {
            if (field.tagName === 'SELECT') {
              field.selectedIndex = 0;
            } else if (field.type !== 'submit' && field.type !== 'button') {
              field.value = '';
            }
          });
        }
      }, 0);
    });
  });
});
