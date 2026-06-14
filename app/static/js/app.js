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

document.addEventListener("submit", async (event) => {
  const form = event.target.closest("[data-review-form]");
  if (!form) {
    return;
  }
  event.preventDefault();
  await saveReviewForm(form);
});

document.addEventListener("submit", async (event) => {
  const form = event.target.closest("[data-adoption-form]");
  if (!form) {
    return;
  }
  event.preventDefault();
  await submitAdoptionForm(form);
});

async function submitAdoptionForm(form) {
  const status = form.querySelector("[data-adoption-status]") || form.querySelector(".save-status");
  const button = form.querySelector("button[type='submit']");
  const originalText = button ? button.textContent : "";
  if (status) {
    status.textContent = "保存中...";
  }
  if (button) {
    button.disabled = true;
  }
  try {
    const response = await fetch(form.action, {
      method: "POST",
      body: new FormData(form),
      headers: { "X-Requested-With": "fetch", "Accept": "application/json" },
    });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    const payload = await response.json();
    applyAdoptionUpdate(payload);
    if (status) {
      status.textContent = payload.message || "保存済み";
    }
    window.setTimeout(() => {
      if (status) {
        status.textContent = "";
      }
    }, 1800);
  } catch (error) {
    if (status) {
      status.textContent = "保存失敗";
    }
    alert(`採用状態の保存に失敗しました: ${error.message}`);
  } finally {
    if (button) {
      button.disabled = false;
      button.textContent = originalText;
    }
  }
}

function applyAdoptionUpdate(payload) {
  if (payload.output_id) {
    document.querySelectorAll("[data-output-selected]").forEach((cell) => {
      cell.textContent = "";
    });
    document.querySelectorAll("[data-output-row]").forEach((row) => {
      row.classList.remove("epoch-selected");
    });
    const selectedRow = document.querySelector(`[data-output-row="${payload.output_id}"]`);
    if (selectedRow) {
      selectedRow.classList.add("epoch-selected");
      const selectedCell = selectedRow.querySelector("[data-output-selected]");
      if (selectedCell) {
        selectedCell.textContent = "採用中";
      }
    }
  }
  const adoptedPath = document.querySelector("[data-adopted-model-path]");
  if (adoptedPath && payload.file_path) {
    adoptedPath.textContent = payload.file_path;
  }
}

async function saveReviewForm(form) {
  const status = form.querySelector(".save-status");
  const button = form.querySelector("button[type='submit']");
  if (status) {
    status.textContent = "保存中...";
  }
  if (button) {
    button.disabled = true;
  }
  try {
    const response = await fetch(form.action, {
      method: "POST",
      body: new FormData(form),
      headers: { "X-Requested-With": "fetch", "Accept": "application/json" },
    });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    if (status) {
      status.textContent = "保存済み";
    }
    form.closest("[data-sample-card]")?.classList.add("review-saved");
  } catch (error) {
    if (status) {
      status.textContent = "保存失敗";
    }
    alert(`評価保存に失敗しました: ${error.message}`);
  } finally {
    if (button) {
      button.disabled = false;
    }
  }
}

document.addEventListener("click", async (event) => {
  const saveAll = event.target.closest("[data-save-all-ratings]");
  if (!saveAll) {
    return;
  }
  event.preventDefault();
  const cards = [...document.querySelectorAll("[data-sample-card]")];
  const items = cards.map((card) => {
    const form = card.querySelector("[data-review-form]");
    const data = new FormData(form);
    const item = { id: card.getAttribute("data-sample-id"), failure_tags: data.getAll("failure_tags") };
    for (const [key, value] of data.entries()) {
      if (key !== "failure_tags") {
        item[key] = value;
      }
    }
    return item;
  });
  saveAll.disabled = true;
  const originalText = saveAll.textContent;
  saveAll.textContent = "一括保存中...";
  try {
    const jobId = location.pathname.match(/\/jobs\/(\d+)/)?.[1];
    const response = await fetch(`/jobs/${jobId}/samples/review-bulk`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "application/json" },
      body: JSON.stringify({ items }),
    });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    const payload = await response.json();
    saveAll.textContent = `${payload.updated}件保存済み`;
    cards.forEach((card) => card.classList.add("review-saved"));
    window.setTimeout(() => { saveAll.textContent = originalText; }, 1800);
  } catch (error) {
    saveAll.textContent = "保存失敗";
    alert(`一括保存に失敗しました: ${error.message}`);
  } finally {
    window.setTimeout(() => {
      saveAll.disabled = false;
      if (saveAll.textContent === "保存失敗") {
        saveAll.textContent = originalText;
      }
    }, 1800);
  }
});

document.addEventListener("click", (event) => {
  const image = event.target.closest("[data-lightbox-src]");
  if (!image) {
    return;
  }
  event.preventDefault();
  openLightbox(image.getAttribute("data-lightbox-src"), image.getAttribute("data-lightbox-title") || "");
});

function openLightbox(src, title) {
  let box = document.querySelector(".lightbox");
  if (!box) {
    box = document.createElement("div");
    box.className = "lightbox";
    box.innerHTML = `
      <div class="lightbox-inner">
        <div class="lightbox-actions">
          <span class="lightbox-title"></span>
          <a class="button" target="_blank" rel="noopener">Open original</a>
          <button type="button" data-lightbox-close>閉じる</button>
        </div>
        <img alt="">
      </div>`;
    document.body.appendChild(box);
    box.addEventListener("click", (event) => {
      if (event.target === box || event.target.closest("[data-lightbox-close]")) {
        box.classList.remove("open");
      }
    });
  }
  box.querySelector("img").src = src;
  box.querySelector("a").href = src;
  box.querySelector(".lightbox-title").textContent = title;
  box.classList.add("open");
}
