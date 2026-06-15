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

document.addEventListener("submit", (event) => {
  const form = event.target.closest("[data-embedding-job-form]");
  if (!form) {
    return;
  }
  const button = form.querySelector("button[type='submit']");
  if (button) {
    button.disabled = true;
    button.dataset.originalText = button.textContent;
    button.textContent = "処理開始中...";
  }
  document.querySelectorAll("[data-embedding-job-form] button[type='submit']").forEach((otherButton) => {
    otherButton.disabled = true;
  });
});

document.addEventListener("submit", async (event) => {
  const form = event.target.closest("[data-reference-candidate-form]");
  if (!form) {
    return;
  }
  event.preventDefault();
  await addReferenceCandidate(form);
});

document.addEventListener("submit", async (event) => {
  const form = event.target.closest("[data-reference-image-delete-form]");
  if (!form) {
    return;
  }
  event.preventDefault();
  await deleteReferenceImage(form);
});

function initEmbeddingJobStatusPolling() {
  const panel = document.querySelector("[data-embedding-job-status]");
  if (!panel) {
    return;
  }
  const jobId = panel.getAttribute("data-embedding-job-id");
  const statusText = panel.querySelector("[data-embedding-job-status-text]");
  if (!jobId || !statusText) {
    return;
  }

  const poll = async () => {
    try {
      const response = await fetch(`/embeddings/jobs/${jobId}/status`, {
        headers: { "Accept": "application/json" },
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = await response.json();
      const total = payload.total_count ?? 0;
      const processed = payload.processed_count ?? 0;
      if (payload.status === "completed") {
        statusText.textContent = `完了: ${payload.ready_count ?? processed} / ${total} 件`;
        panel.classList.remove("warning");
        window.setTimeout(() => {
          const url = new URL(window.location.href);
          url.searchParams.delete("embedding_message");
          url.searchParams.delete("embedding_job_id");
          url.searchParams.delete("embedding_error");
          window.location.href = url.toString();
        }, 900);
        return true;
      }
      if (payload.status === "failed") {
        statusText.textContent = `失敗: ${payload.error_message || ""}`;
        panel.classList.add("warning");
        return true;
      }
      if (payload.status === "stopped") {
        statusText.textContent = `停止: ${processed} / ${total} 件`;
        panel.classList.add("warning");
        return true;
      }
      statusText.textContent = `処理中: ${processed} / ${total} 件`;
      return false;
    } catch (error) {
      statusText.textContent = `状態確認に失敗: ${error.message}`;
      panel.classList.add("warning");
      return true;
    }
  };

  poll().then((done) => {
    if (done) {
      return;
    }
    const timer = window.setInterval(async () => {
      const donePolling = await poll();
      if (donePolling) {
        window.clearInterval(timer);
      }
    }, 2500);
  });
}

async function addReferenceCandidate(form) {
  const button = form.querySelector("button[type='submit']");
  const row = form.closest("tr");
  const originalText = button ? button.textContent : "";
  if (button) {
    button.disabled = true;
    button.textContent = "追加中...";
  }
  try {
    const response = await fetch(form.action, {
      method: "POST",
      body: new FormData(form),
      headers: { "X-Requested-With": "fetch", "Accept": "application/json" },
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || payload.ok === false) {
      throw new Error(payload.message || await response.text());
    }
    row?.classList.add("reference-candidate-added");
    appendReferenceImageCard(payload, form);
    updateReferenceCompleteness(payload.completeness);
    const select = form.querySelector("select");
    if (select) {
      select.disabled = true;
    }
    if (button) {
      button.textContent = "追加済み";
      button.disabled = true;
    }
  } catch (error) {
    if (button) {
      button.disabled = false;
      button.textContent = originalText || "追加";
    }
    alert(`Reference画像の追加に失敗しました: ${error.message}`);
  }
}

function appendReferenceImageCard(payload, form) {
  const grid = document.querySelector("[data-reference-image-grid]");
  if (!grid || !payload.image_id || !payload.image_url) {
    return;
  }
  grid.querySelector("p.muted")?.remove();
  if (grid.querySelector(`[data-reference-image-card="${payload.image_id}"]`)) {
    return;
  }

  const selectedOption = form.querySelector("select[name='image_role'] option:checked");
  const roleLabel = selectedOption ? selectedOption.textContent : (payload.image_role || "-");
  const figure = document.createElement("figure");
  figure.className = "reference-image-new";
  figure.setAttribute("data-reference-image-card", payload.image_id);

  const img = document.createElement("img");
  img.src = payload.image_url;
  img.alt = `Reference #${payload.image_id}`;
  img.setAttribute("data-lightbox-src", payload.image_url);
  img.setAttribute("data-lightbox-title", `Reference #${payload.image_id}`);

  const caption = document.createElement("figcaption");
  caption.className = "reference-caption";
  const sizeText = `${payload.width || "-"} x ${payload.height || "-"} / ${payload.file_size || "-"} bytes`;
  caption.append(
    document.createTextNode(`#${payload.image_id} / ${roleLabel}`),
    document.createElement("br"),
    document.createTextNode(sizeText),
    document.createElement("br"),
    document.createTextNode(payload.caption || "")
  );

  const deleteForm = document.createElement("form");
  deleteForm.method = "post";
  deleteForm.action = `/reference-images/${payload.image_id}/delete`;
  deleteForm.setAttribute("data-reference-image-delete-form", "");
  const referenceSetInput = document.querySelector("[data-reference-set-id]");
  const hidden = document.createElement("input");
  hidden.type = "hidden";
  hidden.name = "reference_set_id";
  hidden.value = referenceSetInput ? referenceSetInput.value : "";
  const deleteButton = document.createElement("button");
  deleteButton.type = "submit";
  deleteButton.className = "danger";
  deleteButton.textContent = "取り消し";
  deleteForm.append(hidden, deleteButton);

  figure.append(img, caption, deleteForm);
  grid.prepend(figure);
}

async function deleteReferenceImage(form) {
  const button = form.querySelector("button[type='submit']");
  const figure = form.closest("[data-reference-image-card]");
  const originalText = button ? button.textContent : "";
  if (button) {
    button.disabled = true;
    button.textContent = "取り消し中...";
  }
  try {
    const response = await fetch(form.action, {
      method: "POST",
      body: new FormData(form),
      headers: { "X-Requested-With": "fetch", "Accept": "application/json" },
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || payload.ok === false) {
      throw new Error(payload.message || await response.text());
    }
    figure?.remove();
    updateReferenceCompleteness(payload.completeness);
  } catch (error) {
    if (button) {
      button.disabled = false;
      button.textContent = originalText || "取り消し";
    }
    alert(`Reference画像の取り消しに失敗しました: ${error.message}`);
  }
}

function updateReferenceCompleteness(completeness) {
  if (!completeness) {
    return;
  }
  const label = document.querySelector("[data-reference-completeness-label]");
  if (label) {
    const text = completeness.label || "UNKNOWN";
    label.textContent = text;
    label.className = `label ${text.toLowerCase()}`;
  }
  const message = document.querySelector("[data-reference-completeness-message]");
  if (message) {
    message.textContent = completeness.message || "-";
  }
  const tbody = document.querySelector("[data-reference-role-counts]");
  if (!tbody) {
    return;
  }
  tbody.textContent = "";
  const roles = completeness.roles || [];
  if (!roles.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 2;
    cell.className = "muted";
    cell.textContent = "役割はまだ登録されていません。";
    row.append(cell);
    tbody.append(row);
    return;
  }
  roles.forEach((role) => {
    const row = document.createElement("tr");
    const name = document.createElement("td");
    const count = document.createElement("td");
    name.textContent = role.label || role.role || "-";
    count.textContent = role.count ?? 0;
    row.append(name, count);
    tbody.append(row);
  });
}

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

function initValidationGenerationPolling() {
  const rows = [...document.querySelectorAll("[data-validation-run-row]")];
  const activeRows = rows.filter((row) => row.getAttribute("data-generation-status") === "running");
  if (!activeRows.length) {
    return;
  }

  const pollRow = async (row) => {
    const runId = row.getAttribute("data-validation-run-row");
    if (!runId || row.getAttribute("data-generation-status") !== "running") {
      return;
    }
    try {
      const payload = await fetchValidationGenerationStatus(runId);
      applyValidationGenerationStatus(row, payload);
    } catch (error) {
      const logRow = document.querySelector(`[data-validation-run-log-row="${runId}"]`);
      const preview = logRow?.querySelector("[data-generation-log-preview]");
      if (preview) {
        preview.textContent = `状態更新に失敗しました: ${error.message}`;
      }
    }
  };

  activeRows.forEach((row) => pollRow(row));
  const timer = window.setInterval(() => {
    const runningRows = [...document.querySelectorAll('[data-validation-run-row][data-generation-status="running"]')];
    if (!runningRows.length) {
      window.clearInterval(timer);
      return;
    }
    runningRows.forEach((row) => pollRow(row));
  }, 5000);
}

async function fetchValidationGenerationStatus(runId) {
  const response = await fetch(`/validation-runs/${runId}/generation/status`, {
    headers: { "Accept": "application/json" },
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

function generationStatusLabel(status, fallback = "-") {
  const labels = {
    queued: "待機中",
    running: "実行中",
    completed: "完了",
    failed: "失敗",
    stopped: "停止",
  };
  return labels[status] || fallback || status || "-";
}

function syncGenerationButtons(container, status) {
  const isRunning = status === "running";
  const runButton = container.querySelector("[data-generation-run-button]");
  const stopButton = container.querySelector("[data-generation-stop-button]");
  if (runButton) {
    runButton.disabled = isRunning;
    runButton.hidden = isRunning;
    if (!isRunning) {
      runButton.removeAttribute("title");
    }
  }
  if (stopButton) {
    stopButton.disabled = !isRunning;
    stopButton.hidden = !isRunning;
  }
}

function syncAllGenerationRunButtons() {
  const isAnyRunning = Boolean(document.querySelector('[data-validation-run-row][data-generation-status="running"]'));
  document.querySelectorAll("[data-validation-run-row]").forEach((row) => {
    const isThisRunning = row.getAttribute("data-generation-status") === "running";
    const runButton = row.querySelector("[data-generation-run-button]");
    if (!runButton || isThisRunning) {
      return;
    }
    runButton.disabled = isAnyRunning;
    if (isAnyRunning) {
      runButton.setAttribute("title", "他の検証画像生成が実行中です");
    } else {
      runButton.removeAttribute("title");
    }
  });
}

function applyValidationGenerationStatus(row, payload) {
  row.setAttribute("data-generation-status", payload.status || "");
  const label = row.querySelector("[data-generation-status-label]");
  if (label) {
    label.className = payload.status ? `label ${payload.status}` : "";
    label.textContent = generationStatusLabel(payload.status, payload.status_label);
  }
  const actual = row.querySelector("[data-validation-actual]");
  const expected = row.querySelector("[data-validation-expected]");
  if (actual) actual.textContent = payload.actual_image_count ?? "0";
  if (expected) expected.textContent = payload.expected_image_count ?? "-";

  const logRow = document.querySelector(`[data-validation-run-log-row="${payload.run_id}"]`);
  if (logRow) {
    const fileCount = logRow.querySelector("[data-generation-file-count]");
    const logSize = logRow.querySelector("[data-generation-log-size]");
    const logUpdated = logRow.querySelector("[data-generation-log-updated]");
    const pid = logRow.querySelector("[data-generation-pid]");
    const preview = logRow.querySelector("[data-generation-log-preview]");
    if (fileCount) fileCount.textContent = payload.file_count ?? "0";
    if (logSize) logSize.textContent = payload.log_size ?? "0";
    if (logUpdated) logUpdated.textContent = payload.log_updated_at || "-";
    if (pid) pid.textContent = payload.process_id || "-";
    if (preview) {
      preview.classList.toggle("empty", !payload.log_preview);
      preview.textContent = payload.log_preview || "ログはまだありません。生成ファイル数が増えていれば処理は進行中です。";
    }
  }
  syncGenerationButtons(row, payload.status || "");
  syncAllGenerationRunButtons();
}

function initValidationGenerationDetailPolling() {
  const panel = document.querySelector("[data-validation-run-detail]");
  if (!panel || panel.getAttribute("data-generation-status") !== "running") {
    return;
  }
  const runId = panel.getAttribute("data-validation-run-detail");
  const pollDetail = async () => {
    if (!runId || panel.getAttribute("data-generation-status") !== "running") {
      return;
    }
    try {
      const payload = await fetchValidationGenerationStatus(runId);
      applyValidationGenerationDetailStatus(panel, payload);
    } catch (error) {
      const preview = panel.querySelector("[data-detail-generation-log-preview]");
      if (preview) {
        preview.textContent = `状態更新に失敗しました: ${error.message}`;
      }
    }
  };

  pollDetail();
  const timer = window.setInterval(() => {
    if (panel.getAttribute("data-generation-status") !== "running") {
      window.clearInterval(timer);
      return;
    }
    pollDetail();
  }, 5000);
}

function applyValidationGenerationDetailStatus(panel, payload) {
  panel.setAttribute("data-generation-status", payload.status || "");
  const statusLabel = panel.querySelector("[data-detail-generation-status-label]");
  if (statusLabel) {
    statusLabel.className = payload.status ? `label ${payload.status}` : "";
    statusLabel.textContent = generationStatusLabel(payload.status, payload.status_label);
  }
  const statusText = panel.querySelector("[data-detail-generation-status-text]");
  if (statusText) statusText.textContent = generationStatusLabel(payload.status, payload.status_label);
  const pid = panel.querySelector("[data-detail-generation-pid]");
  if (pid) pid.textContent = payload.process_id || "-";
  const process = panel.querySelector("[data-detail-generation-process]");
  if (process) {
    if (payload.status === "running") {
      process.innerHTML = payload.process_alive
        ? '<span class="label running">動作中</span>'
        : '<span class="label warning">未確認 / 停止の可能性</span>';
    } else {
      process.textContent = "-";
    }
  }
  const fileCount = panel.querySelector("[data-detail-generation-file-count]");
  const noteFileCount = panel.querySelector("[data-detail-generation-note-file-count]");
  const logSize = panel.querySelector("[data-detail-generation-log-size]");
  const logUpdated = panel.querySelector("[data-detail-generation-log-updated]");
  const noteLogUpdated = panel.querySelector("[data-detail-generation-note-log-updated]");
  const preview = panel.querySelector("[data-detail-generation-log-preview]");
  if (fileCount) fileCount.textContent = payload.file_count ?? "0";
  if (noteFileCount) noteFileCount.textContent = payload.file_count ?? "0";
  if (logSize) logSize.textContent = payload.log_size ?? "0";
  if (logUpdated) logUpdated.textContent = payload.log_updated_at || "-";
  if (noteLogUpdated) noteLogUpdated.textContent = payload.log_updated_at || "-";
  if (preview) {
    preview.classList.toggle("empty", !payload.log_preview);
    preview.textContent = payload.log_preview || "生成ログはまだありません。生成済みファイル数が増えていれば処理は進行中です。";
  }
  syncGenerationButtons(panel, payload.status || "");
}

document.addEventListener("DOMContentLoaded", initValidationGenerationPolling);
document.addEventListener("DOMContentLoaded", initValidationGenerationDetailPolling);
document.addEventListener("DOMContentLoaded", initEmbeddingJobStatusPolling);
