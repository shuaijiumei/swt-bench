// making specific table sortable
function makeSortable(table) {
  const headers = table.querySelectorAll("th");

  tbody = table.querySelector("tbody");
  function sortByIndex(index) {
      sortTableByColumn(tbody, index);
      updateArrows(headers, index);
 };
  headers.forEach((header, index) => {
    header.addEventListener("click", () => sortByIndex(index));
  });

  function sortTableByColumn(table, columnIndex) {
    const rows = Array.from(table.querySelectorAll("tr"));
    const isAscending = table.getAttribute("data-sort-asc") === "true";
    const direction = isAscending ? 1 : -1;

    rows.sort((rowA, rowB) => {
      const cellA = rowA.children[columnIndex].textContent.trim();
      const cellB = rowB.children[columnIndex].textContent.trim();

      const cellANum = parseFloat(cellA);
      const cellBNum = parseFloat(cellB);

      if (!isNaN(cellANum) && !isNaN(cellBNum)) {
        return direction * (cellANum - cellBNum);
      } else {
        return direction * cellA.localeCompare(cellB);
      }
    });

    rows.forEach(row => table.appendChild(row));
    table.setAttribute("data-sort-asc", !isAscending);
  }

  function updateArrows(headers, sortedColumnIndex) {
    headers.forEach((header, index) => {
      const arrow = header.querySelector(".sort-arrow");
      arrow.setAttribute("act-column", index == sortedColumnIndex);
      if (!arrow) {
        return;
      }
      if (index === sortedColumnIndex) {
        const isAscending = tbody.getAttribute("data-sort-asc") === "true";
        arrow.textContent = isAscending ? "\xa0\xa0🔼" : "\xa0\xa0🔽";
      } else {

        arrow.textContent = "️\xa0\xa0\xa0";
      }
    });
  }
  sortByIndex(1);
}
document.addEventListener("DOMContentLoaded", () => {
  makeSortable(document.getElementById("leaderboard-table"));
});