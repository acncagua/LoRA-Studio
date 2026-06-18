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

document.addEventListener("submit", async (event) => {
  const form = event.target.closest("[data-embedding-job-form]");
  if (!form) {
    return;
  }
  event.preventDefault();
  const button = form.querySelector("button[type='submit']");
  const buttons = Array.from(document.querySelectorAll("[data-embedding-job-form] button[type='submit']"));
  if (button) {
    button.disabled = true;
    button.dataset.originalText = button.textContent;
    button.textContent = "処理開始中...";
  }
  buttons.forEach((otherButton) => {
    otherButton.disabled = true;
  });
  clearQueryNoticeParams(["embedding_error"]);
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
    if (payload.embedding_job_id) {
      startEmbeddingJobPolling(payload.embedding_job_id, payload.message || "Embeddingジョブを開始しました。", payload.redirect_url || window.location.href, form);
      return;
    }
    showPageNotice(payload.message || "Embeddingジョブを開始しました。", "info", form);
  } catch (error) {
    showPageNotice(error.message || "Embeddingジョブを開始できませんでした。", "warning", form);
    buttons.forEach((otherButton) => {
      otherButton.disabled = false;
      if (otherButton.dataset.originalText) {
        otherButton.textContent = otherButton.dataset.originalText;
      }
    });
  }
});

document.addEventListener("submit", async (event) => {
  const form = event.target.closest("[data-review-preparation-run-form]");
  if (!form) {
    return;
  }
  event.preventDefault();
  const button = form.querySelector("[data-review-preparation-run-button]") || form.querySelector("button[type='submit']");
  if (button) {
    button.disabled = true;
    button.dataset.originalText = button.textContent;
    button.textContent = "開始中...";
  }
  updateReviewPreparationInline({
    status: "starting",
    message: "レビュー準備を開始しています。二重実行を避けるためボタンを無効化しました。",
    log_tail: "レビュー準備を開始しています。sd-scripts起動後にログが更新されます。",
  }, form);
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
    showPageNotice(payload.message || "レビュー準備を開始しました。", "info", form);
    if (payload.review_session_id) {
      startReviewPreparationPolling(payload.review_session_id, form);
    }
  } catch (error) {
    showPageNotice(error.message || "レビュー準備を開始できませんでした。", "warning", form);
    if (button) {
      button.disabled = false;
      if (button.dataset.originalText) {
        button.textContent = button.dataset.originalText;
      }
    }
  }
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
  pollEmbeddingJobStatus(panel, jobId, statusText);
}

function pollEmbeddingJobStatus(panel, jobId, statusText) {
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
        panel.classList.add("success");
        enableEmbeddingButtons();
        window.setTimeout(async () => {
          const updated = await refreshMachineReviewReadiness();
          if (!updated) {
            reloadAfterEmbeddingJob(panel);
          }
        }, 300);
        return true;
      }
      if (payload.status === "failed") {
        statusText.textContent = `失敗: ${payload.error_message || ""}`;
        panel.classList.add("warning");
        enableEmbeddingButtons();
        return true;
      }
      if (payload.status === "stopped") {
        statusText.textContent = `停止: ${processed} / ${total} 件`;
        panel.classList.add("warning");
        enableEmbeddingButtons();
        return true;
      }
      statusText.textContent = `処理中: ${processed} / ${total} 件`;
      return false;
    } catch (error) {
      statusText.textContent = `状態確認に失敗: ${error.message}`;
      panel.classList.add("warning");
      enableEmbeddingButtons();
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

function reloadAfterEmbeddingJob(panel) {
  const target = panel.getAttribute("data-embedding-reload-url") || window.location.href;
  const url = new URL(target, window.location.href);
  url.searchParams.delete("embedding_message");
  url.searchParams.delete("embedding_job_id");
  url.searchParams.delete("embedding_error");
  if (!url.hash) {
    sessionStorage.setItem("loraStudioRestoreScrollY", String(window.scrollY || 0));
  }
  window.location.href = url.toString();
}

function restoreScrollAfterInlineRefresh() {
  const value = sessionStorage.getItem("loraStudioRestoreScrollY");
  if (!value) {
    return;
  }
  sessionStorage.removeItem("loraStudioRestoreScrollY");
  const y = Number.parseInt(value, 10);
  if (Number.isFinite(y) && y > 0) {
    window.setTimeout(() => window.scrollTo(0, y), 0);
  }
}

function startEmbeddingJobPolling(jobId, message, redirectUrl, anchorForm = null) {
  clearQueryNoticeParams(["embedding_error"]);
  const container = anchorForm?.closest("section") || document.querySelector("main.content") || document.body;
  let panel = container.querySelector("[data-embedding-job-status]");
  if (!panel) {
    panel = document.createElement("p");
    panel.className = "notice";
    panel.setAttribute("data-embedding-job-status", "1");
    const actions = anchorForm?.closest(".actions");
    if (actions) {
      actions.before(panel);
    } else {
      container.prepend(panel);
    }
  }
  panel.classList.remove("warning", "success");
  panel.setAttribute("data-embedding-job-id", jobId);
  if (redirectUrl) {
    panel.setAttribute("data-embedding-reload-url", redirectUrl);
  }
  panel.textContent = "";
  panel.append(document.createTextNode(`${message} `));
  const statusText = document.createElement("span");
  statusText.setAttribute("data-embedding-job-status-text", "1");
  statusText.textContent = "処理中...";
  panel.append(statusText);
  pollEmbeddingJobStatus(panel, jobId, statusText);
}

function enableEmbeddingButtons() {
  document.querySelectorAll("[data-embedding-job-form] button[type='submit']").forEach((button) => {
    button.disabled = false;
    if (button.dataset.originalText) {
      button.textContent = button.dataset.originalText;
    }
  });
}

function coverageText(coverage) {
  if (!coverage) {
    return "0 / 0";
  }
  return `${coverage.ready ?? 0} / ${coverage.total ?? 0}`;
}

function roleText(distribution) {
  const roles = distribution?.roles || {};
  const parts = Object.entries(roles).map(([role, count]) => `${role}:${count}`);
  return parts.length ? parts.join(", ") : "-";
}

function updateReadinessList(list, items) {
  if (!list) {
    return;
  }
  list.innerHTML = "";
  const values = Array.isArray(items) ? items : [];
  list.hidden = values.length === 0;
  values.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    list.appendChild(li);
  });
}

function setReadinessField(container, name, value) {
  const field = container.querySelector(`[data-readiness-field="${name}"]`);
  if (field) {
    field.textContent = value;
  }
}

async function refreshMachineReviewReadiness() {
  const container = document.querySelector("[data-machine-review-readiness]");
  if (!container) {
    return false;
  }
  const url = container.getAttribute("data-readiness-url");
  if (!url) {
    return false;
  }
  try {
    const response = await fetch(url, { headers: { "Accept": "application/json" } });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    const payload = await response.json();
    setReadinessField(container, "provider", payload.provider || "-");
    setReadinessField(container, "reference_count", payload.reference_count ?? 0);
    setReadinessField(container, "reference_roles", roleText(payload.reference_role_distribution));
    setReadinessField(container, "reference_coverage", coverageText(payload.reference_coverage));
    setReadinessField(container, "dataset_coverage", coverageText(payload.dataset_coverage));
    setReadinessField(container, "target_coverage", coverageText(payload.target_coverage));
    setReadinessField(container, "score_coverage", coverageText(payload.score_coverage));
    updateReadinessList(container.querySelector("[data-readiness-warnings]"), payload.warnings);
    updateReadinessList(container.querySelector("[data-readiness-actions]"), payload.next_actions);
    return true;
  } catch (error) {
    showPageNotice(`準備状況の更新に失敗しました: ${error.message}`, "warning");
    return false;
  }
}

function showPageNotice(message, kind = "info", anchorForm = null) {
  if (!message) {
    return;
  }
  const container = anchorForm?.closest("section") || document.querySelector("main.content") || document.body;
  let notice = container.querySelector("[data-js-page-notice]");
  if (!notice) {
    notice = document.createElement("p");
    notice.setAttribute("data-js-page-notice", "1");
    const actions = anchorForm?.closest(".actions");
    if (actions) {
      actions.before(notice);
    } else {
      container.prepend(notice);
    }
  }
  notice.className = kind === "warning" ? "notice warning" : "notice";
  notice.textContent = message;
}

function startReviewPreparationPolling(sessionId, anchorForm = null) {
  const jobMatch = window.location.pathname.match(/\/jobs\/(\d+)/);
  const jobId = jobMatch ? jobMatch[1] : null;
  if (!jobId || !sessionId) {
    return;
  }
  const poll = async () => {
    try {
      const response = await fetch(`/jobs/${jobId}/review-sessions/${sessionId}/status`, {
        headers: { "Accept": "application/json" },
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = await response.json();
      updateReviewPreparationInline(payload, anchorForm);
      if (["completed", "failed", "stopped"].includes(payload.status)) {
        return true;
      }
      return false;
    } catch (error) {
      showPageNotice(`レビュー準備の状態確認に失敗: ${error.message}`, "warning", anchorForm);
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

function updateReviewPreparationInline(payload, anchorForm = null) {
  const section = anchorForm?.closest("#review-preparation") || document.querySelector("#review-preparation") || document.body;
  const status = section.querySelector("[data-review-preparation-status]");
  if (status && payload.status) {
    status.textContent = payload.status;
  }
  const images = section.querySelector("[data-review-preparation-images]");
  if (images) {
    const liveGenerated = payload.live_generated_image_count ?? payload.generated_image_count;
    if (payload.imported_image_count !== undefined && payload.expected_image_count !== undefined) {
      images.textContent = `${payload.imported_image_count} / ${payload.expected_image_count}（生成 ${liveGenerated ?? 0}）`;
    }
  }
  const scores = section.querySelector("[data-review-preparation-scores]");
  if (scores && payload.scored_image_count !== undefined && payload.expected_image_count !== undefined) {
    scores.textContent = `${payload.scored_image_count} / ${payload.expected_image_count}`;
  }
  const logSize = section.querySelector("[data-review-preparation-log-size]");
  if (logSize && payload.log_size !== undefined) {
    logSize.textContent = payload.log_size;
  }
  const log = section.querySelector("[data-review-preparation-log]");
  if (log) {
    log.classList.remove("empty");
    log.textContent = payload.log_tail || payload.message || "処理開始待ちです。sd-scripts起動後にログが更新されます。";
  }
  const runButton = section.querySelector("[data-review-preparation-run-button]");
  if (runButton && ["completed", "failed", "stopped"].includes(payload.status)) {
    runButton.disabled = false;
    runButton.textContent = runButton.dataset.originalText || "候補レビューを開始";
  }
  if (payload.matrix_ready && payload.matrix_url && !section.querySelector(`[href="${payload.matrix_url}"]`)) {
    const actions = section.querySelector(".actions");
    if (actions) {
      const link = document.createElement("a");
      link.className = "button";
      link.href = payload.matrix_url;
      link.textContent = "レビューMatrixを開く";
      actions.appendChild(link);
    }
  }
}

function initReviewPreparationPolling() {
  const section = document.querySelector("#review-preparation[data-review-preparation-session-id]");
  if (!section) {
    return;
  }
  const status = section.getAttribute("data-review-preparation-session-status");
  const sessionId = section.getAttribute("data-review-preparation-session-id");
  if (sessionId && ["running", "starting"].includes(status)) {
    startReviewPreparationPolling(sessionId, section);
  }
}

function clearQueryNoticeParams(keys) {
  const url = new URL(window.location.href);
  let changed = false;
  keys.forEach((key) => {
    if (url.searchParams.has(key)) {
      url.searchParams.delete(key);
      changed = true;
    }
  });
  if (changed) {
    window.history.replaceState({}, "", url.toString());
  }
}

function initMachineReviewJobStatusPolling() {
  const panel = document.querySelector("[data-machine-review-job-status]");
  if (!panel) {
    return;
  }
  const jobId = panel.getAttribute("data-machine-review-job-id");
  const statusText = panel.querySelector("[data-machine-review-job-status-text]");
  if (!jobId || !statusText) {
    return;
  }

  const poll = async () => {
    try {
      const response = await fetch(`/machine-review/jobs/${jobId}/status`, {
        headers: { "Accept": "application/json" },
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = await response.json();
      const total = payload.total_count ?? 0;
      const processed = payload.processed_count ?? 0;
      const scored = payload.scored_count ?? 0;
      const failed = payload.failed_count ?? 0;
      const skipped = payload.skipped_count ?? 0;
      if (payload.status === "completed") {
        statusText.textContent = `完了: スコア ${scored} / ${total} 件`;
        panel.classList.remove("warning");
        window.setTimeout(() => {
          const url = new URL(window.location.href);
          url.searchParams.delete("machine_review_message");
          url.searchParams.delete("machine_review_job_id");
          url.searchParams.delete("machine_review_error");
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
      statusText.textContent = `処理中: ${processed} / ${total} 件（スコア ${scored}, スキップ ${skipped}, 失敗 ${failed}）`;
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
  if (payload.epoch !== undefined && payload.epoch !== null) {
    document.querySelectorAll("[data-review-epoch-row]").forEach((row) => {
      row.classList.remove("epoch-selected");
      row.classList.add("epoch-candidate");
      const selectedCell = row.querySelector("[data-review-output-selected]");
      if (selectedCell) {
        selectedCell.textContent = "-";
      }
      const machineCell = row.querySelector("[data-review-machine-cell]");
      if (machineCell) {
        machineCell.querySelectorAll("[data-human-rating-priority]").forEach((node) => {
          node.remove();
        });
      }
    });
    const selectedReviewRow = document.querySelector(`[data-review-epoch-row="${payload.epoch}"]`);
    if (selectedReviewRow) {
      selectedReviewRow.classList.remove("epoch-candidate");
      selectedReviewRow.classList.add("epoch-selected");
      const selectedCell = selectedReviewRow.querySelector("[data-review-output-selected]");
      if (selectedCell) {
        selectedCell.textContent = "採用中";
      }
      const machineCell = selectedReviewRow.querySelector("[data-review-machine-cell]");
      if (machineCell && !machineCell.querySelector("[data-human-rating-priority]")) {
        const lineBreak = document.createElement("br");
        lineBreak.setAttribute("data-human-rating-priority", "break");
        const note = document.createElement("span");
        note.className = "muted";
        note.setAttribute("data-human-rating-priority", "note");
        note.textContent = "人間評価を優先";
        machineCell.append(lineBreak, note);
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
          <a class="button" target="_blank" rel="noopener">元画像を開く</a>
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

function initTrainLogPolling() {
  const panel = document.querySelector("[data-train-log-panel]");
  if (!panel) {
    return;
  }
  const url = panel.getAttribute("data-train-log-url");
  const tail = panel.querySelector("[data-train-log-tail]");
  const status = panel.querySelector("[data-train-log-status]");
  if (!url || !tail) {
    return;
  }

  const update = async () => {
    try {
      const response = await fetch(url, { headers: { "Accept": "application/json" } });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = await response.json();
      tail.textContent = payload.log_tail || "train.logはまだありません。";
      const size = payload.log_size ?? 0;
      const updated = payload.log_updated_at || "-";
      const pid = payload.process_id || "-";
      if (status) {
        status.textContent = `状態: ${payload.status || "-"} / PID: ${pid} / ログ: ${size} bytes / 更新: ${updated}`;
      }
      panel.setAttribute("data-train-log-running", payload.status === "running" ? "1" : "0");
      return payload.status === "running";
    } catch (error) {
      if (status) {
        status.textContent = `train.log更新確認に失敗: ${error.message}`;
      }
      return false;
    }
  };

  if (panel.getAttribute("data-train-log-running") !== "1") {
    update();
    return;
  }
  update();
  const timer = window.setInterval(async () => {
    const keepPolling = await update();
    if (!keepPolling) {
      window.clearInterval(timer);
    }
  }, 5000);
}

function initActiveOperationMonitorPolling() {
  const panel = document.querySelector("[data-active-operation-monitor]");
  if (!panel) {
    return;
  }
  const url = panel.getAttribute("data-operation-status-url");
  const refreshButton = panel.querySelector("[data-operation-refresh]");
  const statusText = panel.querySelector('[data-operation-field="status"]');
  const pidText = panel.querySelector('[data-operation-field="pid"]');
  const returnCodeText = panel.querySelector('[data-operation-field="return_code"]');
  const progressText = panel.querySelector('[data-operation-field="progress"]');
  const logUpdateText = panel.querySelector('[data-operation-field="last_log_update"]');
  const shortLog = panel.querySelector("[data-operation-log-short]");
  const fullLog = panel.querySelector("[data-operation-log-full]");
  const warning = panel.querySelector("[data-operation-log-warning]");

  const applyPayload = (payload) => {
    if (statusText && payload.status !== undefined) statusText.textContent = payload.status || "-";
    const processId = payload.process_id ?? payload.generation_process_id;
    if (pidText && processId !== undefined) pidText.textContent = processId || "-";
    if (returnCodeText && payload.return_code !== undefined) returnCodeText.textContent = payload.return_code ?? "-";
    const current = payload.current
      ?? payload.processed_count
      ?? payload.generated_image_count
      ?? payload.live_generated_image_count
      ?? payload.imported_image_count
      ?? payload.scored_image_count;
    const total = payload.total
      ?? payload.total_count
      ?? payload.expected_image_count;
    const progress = payload.progress_label
      || (payload.current != null && payload.total != null ? `${payload.current} / ${payload.total}` : "")
      || (current != null && total != null ? `${current} / ${total}` : "")
      || (current != null ? `${current}` : "");
    if (progressText && progress) progressText.textContent = progress;
    const logTail = payload.log_tail || payload.log_preview || payload.generation_log_tail || "";
    if (shortLog && logTail) shortLog.textContent = logTail;
    if (fullLog && logTail) fullLog.textContent = logTail;
    const size = payload.log_size ?? "";
    const updated = payload.log_updated_at || "";
    if (logUpdateText && (size !== "" || updated)) {
      logUpdateText.textContent = `${updated || "-"}${size !== "" ? ` / ${size} bytes` : ""}`;
    }
    if (warning && payload.log_warning) {
      warning.hidden = false;
      warning.textContent = payload.log_warning;
    }
    if (warning && !payload.log_warning) {
      warning.hidden = true;
      warning.textContent = "";
    }
    const runningStatuses = new Set(["running", "generating_images", "embedding_images", "machine_reviewing", "building_matrix"]);
    const isRunning = runningStatuses.has(payload.status);
    panel.setAttribute("data-operation-running", isRunning ? "1" : "0");
    return isRunning;
  };

  const update = async () => {
    if (!url) {
      return false;
    }
    try {
      const response = await fetch(url, { headers: { "Accept": "application/json" } });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      return applyPayload(await response.json());
    } catch (error) {
      if (warning) {
        warning.hidden = false;
        warning.textContent = `状態更新に失敗: ${error.message}`;
      }
      return false;
    }
  };

  if (refreshButton) {
    refreshButton.addEventListener("click", async () => {
      if (url) {
        await update();
      } else {
        window.location.reload();
      }
    });
  }
  if (!url || panel.getAttribute("data-operation-running") !== "1") {
    return;
  }
  update();
  const timer = window.setInterval(async () => {
    const keepPolling = await update();
    if (!keepPolling) {
      window.clearInterval(timer);
    }
  }, 5000);
}

document.addEventListener("DOMContentLoaded", initValidationGenerationPolling);
document.addEventListener("DOMContentLoaded", initValidationGenerationDetailPolling);
document.addEventListener("DOMContentLoaded", restoreScrollAfterInlineRefresh);
document.addEventListener("DOMContentLoaded", initEmbeddingJobStatusPolling);
document.addEventListener("DOMContentLoaded", initMachineReviewJobStatusPolling);
document.addEventListener("DOMContentLoaded", initReviewPreparationPolling);
document.addEventListener("DOMContentLoaded", initTrainLogPolling);
document.addEventListener("DOMContentLoaded", initActiveOperationMonitorPolling);
