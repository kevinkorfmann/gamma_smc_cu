document.addEventListener("DOMContentLoaded", () => {
  const input = document.querySelector("#locus-search");
  if (!input) return;
  const rows = [...document.querySelectorAll("#locus-table tbody tr")];
  const count = document.querySelector("#locus-count");
  input.addEventListener("input", () => {
    const terms = input.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
    let visible = 0;
    rows.forEach(row => {
      row.hidden = !terms.every(term => row.dataset.search.toLowerCase().includes(term));
      if (!row.hidden) visible++;
    });
    count.textContent = `${visible} of ${rows.length} loci`;
  });
});
