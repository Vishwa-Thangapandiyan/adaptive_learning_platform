document.addEventListener("DOMContentLoaded", () => {
  const toggles = document.querySelectorAll(".collapse-toggle");
  toggles.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = document.querySelector(btn.dataset.target);
      if (!target) return;
      target.classList.toggle("open");
    });
  });
});
