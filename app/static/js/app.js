document.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-browse-target]");
  if (!button) {
    return;
  }

  event.preventDefault();
  const targetSelector = button.getAttribute("data-browse-target");
  const target = document.querySelector(targetSelector);
  if (!target) {
    return;
  }

  const kind = button.getAttribute("data-browse-kind") || "directory";
  const title = button.getAttribute("data-browse-title") || "選択";
  const initialPath = target.value || button.getAttribute("data-browse-initial") || "";
  const query = new URLSearchParams({ title, initial_path: initialPath });
  const url = kind === "directory"
    ? `/api/browse-directory?${query.toString()}`
    : `/api/browse-file?${new URLSearchParams({ kind, title, initial_path: initialPath }).toString()}`;

  const originalText = button.textContent;
  button.disabled = true;
  button.textContent = "選択中...";
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(await response.text());
    }
    const payload = await response.json();
    if (payload.path) {
      target.value = payload.path;
      target.dispatchEvent(new Event("input", { bubbles: true }));
      target.dispatchEvent(new Event("change", { bubbles: true }));
    }
  } catch (error) {
    alert(`パス選択に失敗しました: ${error.message}`);
  } finally {
    button.disabled = false;
    button.textContent = originalText;
  }
});

document.addEventListener("change", (event) => {
  const select = event.target.closest("[data-fill-target]");
  if (!select) {
    return;
  }
  const target = document.querySelector(select.getAttribute("data-fill-target"));
  if (target && select.value) {
    target.value = select.value;
  }
});
