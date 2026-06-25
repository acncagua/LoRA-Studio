function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

const LORA_STUDIO_LOCALE = (document.documentElement.getAttribute("lang") || "ja").split("-", 1)[0];
const LORA_STUDIO_TEXT = {
  ja: {
    recipeNotSelected: "Recipe未選択",
    chooseByPurpose: "用途から選ぶ",
    chooseByOptimizer: "Optimizerから選ぶ",
    deriveFromJob: "既存Jobから派生",
    fullCustom: "完全カスタム",
    recipeMissingCustom: "Recipe v2未選択です。完全カスタムとして作成します。",
    recipeMissingError: "Training Recipe v2が未選択です。フィルタだけでは設定は反映されません。Recipeカードを選択してください。",
    smokeUntested: "Optimizer ProfileはSmoke Test未検証です。AdamW8bit以外ではOptimizer詳細からSmoke Testを実行してから使うことを推奨します。",
    prepareOnly: "Optimizer ProfileはPrepare TestのみOKです。2-step Smoke Testは未実行です。",
    smokeFailed: "Optimizer Profileの前回Smoke Testが失敗しています。依存関係・optimizer指定・ログを確認してください。",
    disabledProfile: "Optimizer Profileはdisabled扱いです。通常運用では選択しないでください。",
    miniUntested: "Optimizer ProfileはSmoke OKでもMini Pilot未確認です。実用学習では挙動が未確認です。",
    miniFailed: "Optimizer ProfileはMini Pilotで失敗しています。通常選択は推奨されません。",
    miniWarning: "Optimizer ProfileはMini Pilotで警告があります。ログとartifact確認結果を見てください。",
    miniOk: "Mini Pilot OK: 短時間実学習の起動・artifact確認を通過しています。",
    advancedOptimizer: "optimizerです。比較用または上級者向けとして扱ってください。",
    networkUnavailable: "Network Typeは{value}です。現在の実行対象ではありません。",
    cacheTeError: "cache_text_encoder_outputs=true で text_encoder_lr が有効です。TEを学習する場合はcacheを外してください。",
    unetTeError: "network_train_unet_only=true で text_encoder_lr が有効です。UNetのみ学習かTE学習のどちらかに揃えてください。",
    schedulerWarning: "DAdapt / Prodigyではscheduler=constantを推奨します。",
    adafactorNote: "Adafactor relative_step系は通常LRと意味が異なります。profile説明を確認してください。",
    rawArgsObject: "Raw Args JSONはオブジェクトで入力してください。",
    rawArgsParse: "Raw Args JSONを読めません: {message}",
    rawArgsWarning: "Raw Argsが指定されています。Recipeの標準互換性チェック外の項目を含む可能性があります。",
    filterHint: "フィルタは候補の絞り込みです。実際に使うRecipeカードを選択してください。",
    noFilterRecipes: "現在のフィルタ条件に合うRecipeがありません。Optimizer ProfileやNetwork Typeを変更してください。",
    customNoRecipe: "完全カスタムではRecipe未選択のまま作成できます。",
    selectedHint: "左のRecipeカードから選択してください。",
    noRecipeParams: "Recipe未選択です。フィルタだけでは学習パラメータは確定しません。Recipeカードを選択してください。",
    noDiff: "差分はありません。Recipe標準値のままです。",
    recipeValue: "作成値",
    noErrors: "ERRORはありません。",
    noWarnings: "WARNINGはありません。",
    recommended: "おすすめ",
    chooseRecipe: "このRecipeを選択",
    recipeCount: "{count}件",
    optimizerGroupingHint: "選択Optimizerで使えるRecipeを用途別に表示しています。",
    purposeGroupingHint: "用途に合うRecipeをOptimizerカテゴリ別に表示しています。",
    purpose: "用途",
    expectedBehavior: "期待する挙動",
    note: "注意",
    risk: "注意",
    compatibilityNotes: "互換メモ",
    details: "詳細を開く",
    countSuffix: "件",
    select: "選択",
    selecting: "選択中...",
    pathSelectFailed: "パス選択に失敗しました: {message}",
    projectCurrentSelection: "現在の選択: {name} / trigger {trigger}",
    projectSelectExisting: "既存Projectを選択してください。",
    useSuggestion: "この候補を入力",
    reviewStandardAutoNote: "標準自動はStandard Validation v1を候補epochごとに実行します。候補3件なら45枚×3=135枚のため、既定では150枚・240分まで自動開始します。",
    reviewQuickAutoNote: "クイック自動は候補epoch最大3件、prompt 3種、seed 1件、weight 2種の最大18枚を自動生成します。",
    reviewManualNote: "手動ではReview Plan作成も画像生成も自動では行いません。",
    reviewPlanOnlyNote: "計画のみではReview Planだけ作成し、画像生成は自動開始しません。",
  },
  en: {
    recipeNotSelected: "Recipe not selected",
    chooseByPurpose: "Choose by Purpose",
    chooseByOptimizer: "Choose by Optimizer",
    deriveFromJob: "Derive from Existing Job",
    fullCustom: "Full Custom",
    recipeMissingCustom: "No Recipe v2 is selected. A full custom Job will be created.",
    recipeMissingError: "Training Recipe v2 is not selected. Filters do not apply settings by themselves. Select a Recipe card.",
    smokeUntested: "This Optimizer Profile has not passed Smoke Test yet. For non-AdamW8bit optimizers, run Smoke Test from Optimizer details first.",
    prepareOnly: "This Optimizer Profile has only passed Prepare Test. 2-step Smoke Test has not run.",
    smokeFailed: "The previous Smoke Test failed. Check dependencies, optimizer settings, and logs.",
    disabledProfile: "This Optimizer Profile is disabled and should not be used for normal jobs.",
    miniUntested: "This Optimizer Profile has Smoke OK but no Mini Pilot result yet. Practical behavior is unverified.",
    miniFailed: "This Optimizer Profile failed Mini Pilot and is not recommended for normal selection.",
    miniWarning: "This Optimizer Profile has Mini Pilot warnings. Check logs and artifact results.",
    miniOk: "Mini Pilot OK: short real training startup and artifact checks passed.",
    advancedOptimizer: "optimizer. Treat this as comparison or advanced use.",
    networkUnavailable: "Network Type is {value}. It is not currently enabled for execution.",
    cacheTeError: "cache_text_encoder_outputs=true conflicts with enabled text_encoder_lr. Disable cache when training TE.",
    unetTeError: "network_train_unet_only=true conflicts with enabled text_encoder_lr. Choose either UNet-only or TE training.",
    schedulerWarning: "DAdapt / Prodigy should usually use scheduler=constant.",
    adafactorNote: "Adafactor relative_step profiles use LR differently from normal LR. Check the profile description.",
    rawArgsObject: "Raw Args JSON must be an object.",
    rawArgsParse: "Could not parse Raw Args JSON: {message}",
    rawArgsWarning: "Raw Args are set. They may include items outside the standard Recipe compatibility checks.",
    filterHint: "Filters only narrow candidates. Select the Recipe card you will actually use.",
    noFilterRecipes: "No Recipes match the current filters. Change Optimizer Profile or Network Type.",
    customNoRecipe: "Full Custom can be created without selecting a Recipe.",
    selectedHint: "Select a Recipe card on the left.",
    noRecipeParams: "Recipe is not selected. Filters alone do not decide training parameters. Select a Recipe card.",
    noDiff: "No differences. Recipe defaults are unchanged.",
    recipeValue: "Job value",
    noErrors: "No ERRORs.",
    noWarnings: "No WARNINGs.",
    recommended: "Recommended",
    chooseRecipe: "Select this Recipe",
    recipeCount: "{count} recipes",
    optimizerGroupingHint: "Recipes available for the selected optimizer are grouped by purpose.",
    purposeGroupingHint: "Recipes matching the purpose are grouped by optimizer category.",
    purpose: "Purpose",
    expectedBehavior: "Expected behavior",
    note: "Note",
    risk: "Risk",
    compatibilityNotes: "Compatibility notes",
    details: "Open details",
    countSuffix: "",
    select: "Select",
    selecting: "Selecting...",
    pathSelectFailed: "Path selection failed: {message}",
    projectCurrentSelection: "Current selection: {name} / trigger {trigger}",
    projectSelectExisting: "Select an existing Project.",
    useSuggestion: "Use this suggestion",
    reviewStandardAutoNote: "Standard Auto runs Standard Validation v1 for each candidate epoch. With 3 candidates it generates 45 x 3 = 135 images, so the default auto-start limits are 150 images and 240 minutes.",
    reviewQuickAutoNote: "Quick Auto generates up to 18 images: up to 3 candidate epochs, 3 prompts, 1 seed, and 2 weights.",
    reviewManualNote: "Manual mode does not automatically create Review Plans or generate images.",
    reviewPlanOnlyNote: "Plan Only creates only the Review Plan and does not start image generation automatically.",
  },
};

function uiText(key, fallback = "") {
  const table = LORA_STUDIO_TEXT[LORA_STUDIO_LOCALE] || LORA_STUDIO_TEXT.ja;
  return table[key] || LORA_STUDIO_TEXT.en[key] || fallback || key;
}

function uiFormat(key, values = {}, fallback = "") {
  return uiText(key, fallback).replace(/\{(\w+)\}/g, (_, name) => values[name] ?? "");
}

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
  const title = button.getAttribute("data-browse-title") || uiText("select");
  const initialPath = target.value || button.getAttribute("data-browse-initial") || "";
  const query = new URLSearchParams({ title, initial_path: initialPath });
  const url = kind === "directory"
    ? `/api/browse-directory?${query.toString()}`
    : `/api/browse-file?${new URLSearchParams({ kind, title, initial_path: initialPath }).toString()}`;

  const originalText = button.textContent;
  button.disabled = true;
  button.textContent = uiText("selecting");
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
    alert(uiFormat("pathSelectFailed", { message: error.message }));
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

function updateProjectModeForm(form) {
  if (!form) {
    return;
  }
  const mode = form.querySelector("input[name='project_mode']:checked")?.value || "new";
  const fields = form.querySelector("[data-new-project-fields]");
  const existingFields = form.querySelector("[data-existing-project-fields]");
  const note = form.querySelector("[data-existing-project-note]");
  const summary = form.querySelector("[data-existing-project-summary]");
  const projectSelect = form.querySelector("select[name='project_id']");
  const isExisting = mode === "existing";

  if (fields) {
    fields.hidden = isExisting;
  }
  if (existingFields) {
    existingFields.hidden = !isExisting;
  }
  form.querySelectorAll("[data-new-project-field]").forEach((input) => {
    input.disabled = isExisting;
  });
  if (note) {
    note.hidden = !isExisting;
  }
  if (summary && projectSelect) {
    const selected = projectSelect.options[projectSelect.selectedIndex];
    const name = selected?.getAttribute("data-project-name") || "";
    const trigger = selected?.getAttribute("data-project-trigger") || "";
    summary.textContent = name
      ? ` ${uiFormat("projectCurrentSelection", { name, trigger: trigger || "-" })}`
      : ` ${uiText("projectSelectExisting")}`;
  }
}

function initProjectModeForms() {
  document.querySelectorAll("[data-project-mode-form]").forEach((form) => {
    updateProjectModeForm(form);
    form.addEventListener("change", (event) => {
      if (event.target.matches("input[name='project_mode'], select[name='project_id']")) {
        updateProjectModeForm(form);
      }
    });
  });
}

function updateDatasetVersionSelect(datasetSelect) {
  if (!datasetSelect) {
    return;
  }
  const form = datasetSelect.closest("form") || document;
  const versionSelect = form.querySelector("[data-dataset-version-select]");
  if (!versionSelect) {
    return;
  }
  const datasetId = String(datasetSelect.value || "");
  let selectedStillVisible = false;
  Array.from(versionSelect.options).forEach((option) => {
    const optionDatasetId = option.getAttribute("data-dataset-id");
    const visible = !option.value || optionDatasetId === datasetId;
    option.hidden = !visible;
    option.disabled = !visible;
    if (visible && option.selected) {
      selectedStillVisible = true;
    }
  });
  if (!selectedStillVisible) {
    versionSelect.value = "";
  }
}

function initDatasetVersionFilters() {
  document.querySelectorAll("[data-dataset-select]").forEach((datasetSelect) => {
    updateDatasetVersionSelect(datasetSelect);
    datasetSelect.addEventListener("change", () => updateDatasetVersionSelect(datasetSelect));
  });
}

function parseJsonScript(selector, root = document) {
  const node = root.querySelector(selector);
  if (!node) {
    return [];
  }
  try {
    return JSON.parse(node.textContent || "[]");
  } catch (_error) {
    return [];
  }
}

function stepEstimatorParams(form) {
  const presets = parseJsonScript("[data-step-presets-json]", form);
  const recipes = parseJsonScript("[data-recipes-v2-json]", form);
  const recipeId = form.querySelector("select[name='recipe_v2_id']")?.value || "";
  const recipe = recipes.find((item) => item.id === recipeId) || null;
  const presetId = form.querySelector("select[name='preset_id']")?.value || "";
  const preset = presets.find((item) => item.id === presetId) || {};
  const params = { ...((recipe && recipe.params) || preset.params || {}) };
  [
    "repeats",
    "max_train_epochs",
    "train_batch_size",
    "gradient_accumulation_steps",
    "save_every_n_epochs",
    "learning_rate",
    "unet_lr",
    "text_encoder_lr1",
    "text_encoder_lr2",
    "network_dim",
    "network_alpha",
    "optimizer_type",
    "lr_scheduler",
  ].forEach((name) => {
    const input = form.querySelector(`[name='${name}']`);
    if (input && input.value !== "") {
      const value = Number(input.value);
      if (Number.isFinite(value)) {
        params[name] = value;
      } else {
        params[name] = input.value;
      }
    }
  });
  ["cache_latents", "cache_text_encoder_outputs", "network_train_unet_only", "generate_training_samples"].forEach((name) => {
    const input = form.querySelector(`[name='${name}']`);
    if (!input) {
      return;
    }
    if (input.type === "checkbox") {
      params[name] = input.checked;
    } else if (input.value === "1") {
      params[name] = true;
    } else if (input.value === "0") {
      params[name] = false;
    }
  });
  const raw = form.querySelector("[data-param-editor-raw]");
  if (raw && raw.value.trim()) {
    try {
      const rawParams = JSON.parse(raw.value);
      if (rawParams && typeof rawParams === "object" && !Array.isArray(rawParams)) {
        Object.assign(params, rawParams);
      }
    } catch (_error) {
      // Compatibility panel reports invalid JSON; Step Estimate can keep the parsed params.
    }
  }
  return params;
}

async function refreshStepEstimate(form, withSuggestions = false) {
  const panel = form.querySelector("[data-step-estimator]");
  if (!panel) {
    return;
  }
  const payload = {
    dataset_id: form.querySelector("[name='dataset_id']")?.value || "",
    dataset_version_id: form.querySelector("[name='dataset_version_id']")?.value || "",
    preset_id: form.querySelector("[name='preset_id']")?.value || "",
    recipe_v2_id: form.querySelector("[name='recipe_v2_id']")?.value || "",
    params: stepEstimatorParams(form),
    target_steps: panel.querySelector("[data-step-target]")?.value || "",
    strategy: panel.querySelector("[data-step-strategy]")?.value || "balanced",
  };
  try {
    const response = await fetch("/api/step-estimate", {
      method: "POST",
      headers: {"Content-Type": "application/json", "Accept": "application/json"},
      body: JSON.stringify(withSuggestions ? payload : {...payload, target_steps: ""}),
    });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    const data = await response.json();
    updateStepEstimatePanel(panel, data.estimate || {});
    updateSelectedRecipePanel(form);
    if (withSuggestions) {
      applyAutoRepeat(form, data.auto_repeat || {});
      updateStepSuggestions(form, data.suggestions || []);
    }
  } catch (error) {
    const message = panel.querySelector("[data-step-field='message']");
    if (message) {
      message.textContent = `Step Estimateを更新できませんでした: ${error.message}`;
    }
  }
}

function updateStepEstimatePanel(panel, estimate) {
  const flatEstimate = {...estimate};
  if (estimate.duration_estimate) {
    flatEstimate.duration_estimated_label = estimate.duration_estimate.estimated_label || "-";
    flatEstimate.duration_seconds_per_step = estimate.duration_estimate.seconds_per_step ?? "-";
    flatEstimate.duration_basis = estimate.duration_estimate.basis || "";
  }
  Object.entries(flatEstimate).forEach(([key, value]) => {
    const node = panel.querySelector(`[data-step-field='${key}']`);
    if (!node) {
      return;
    }
    if (key === "status") {
      node.textContent = value ?? "-";
      node.className = `label ${String(value || "unknown").toLowerCase()}`;
    } else if (Array.isArray(value)) {
      node.textContent = value.join(" / ");
    } else {
      node.textContent = value ?? "-";
    }
  });
  const warnings = panel.querySelector("[data-step-warnings]");
  if (warnings) {
    warnings.innerHTML = "";
    (estimate.warnings || []).forEach((warning) => {
      const li = document.createElement("li");
      li.textContent = warning;
      warnings.appendChild(li);
    });
  }
  const targetInput = panel.querySelector("[data-step-target]");
  if (targetInput && estimate.target_steps_recommended && targetInput.dataset.manualTarget !== "true") {
    targetInput.value = estimate.target_steps_recommended;
  }
}

function applyAutoRepeat(form, autoRepeat) {
  const panel = form.querySelector("[data-step-estimator]");
  const target = panel?.querySelector("[data-step-suggestions]");
  if (!autoRepeat || !autoRepeat.required_repeats) {
    if (target) {
      target.textContent = autoRepeat?.error || "repeatsを自動計算できませんでした。";
    }
    return;
  }
  const repeats = form.querySelector("[name='repeats']");
  const autoFlag = form.querySelector("[data-repeats-auto-calculated]");
  if (repeats) {
    repeats.value = autoRepeat.required_repeats;
  }
  if (autoFlag) {
    autoFlag.value = "1";
  }
  if (target) {
    target.textContent = `repeats=${autoRepeat.required_repeats} を入力しました。expected_total_steps=${autoRepeat.expected_total_steps} です。`;
  }
  refreshStepEstimate(form);
}

function updateStepSuggestions(form, suggestions) {
  const panel = form.querySelector("[data-step-estimator]");
  const target = panel?.querySelector("[data-step-suggestions]");
  if (!target) {
    return;
  }
  const currentMessage = target.textContent;
  target.innerHTML = "";
  if (currentMessage) {
    const p = document.createElement("p");
    p.textContent = currentMessage;
    target.appendChild(p);
  }
  if (!suggestions.length) {
    return;
  }
  const table = document.createElement("table");
  table.innerHTML = "<thead><tr><th>repeats</th><th>epochs</th><th>steps</th><th>保存間隔</th><th>反映</th></tr></thead><tbody></tbody>";
  const tbody = table.querySelector("tbody");
  suggestions.forEach((item) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${item.repeats}</td><td>${item.max_train_epochs}</td><td>${item.expected_total_steps}</td><td>${item.save_every_n_epochs_proposal}</td><td></td>`;
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = uiText("useSuggestion");
    button.addEventListener("click", () => {
      const repeats = form.querySelector("[name='repeats']");
      const epochs = form.querySelector("[name='max_train_epochs']");
      const saveEvery = form.querySelector("[name='save_every_n_epochs']");
      if (repeats) repeats.value = item.repeats;
      if (epochs) epochs.value = item.max_train_epochs;
      if (saveEvery) saveEvery.value = item.save_every_n_epochs_proposal;
      refreshStepEstimate(form);
    });
    tr.lastElementChild.appendChild(button);
    tbody.appendChild(tr);
  });
  target.appendChild(table);
}

function initStepEstimators() {
  document.querySelectorAll("form").forEach((form) => {
    if (!form.querySelector("[data-step-estimator]")) {
      return;
    }
    form.addEventListener("input", (event) => {
      if (event.target.matches("[name='repeats'], [name='max_train_epochs'], [name='train_batch_size'], [name='gradient_accumulation_steps'], [name='save_every_n_epochs'], [name='learning_rate'], [name='unet_lr'], [name='text_encoder_lr1'], [name='text_encoder_lr2'], [name='network_dim'], [name='network_alpha'], [name='optimizer_type'], [name='lr_scheduler'], [data-param-editor-raw]")) {
        if (event.target.matches("[name='repeats']")) {
          const autoFlag = form.querySelector("[data-repeats-auto-calculated]");
          if (autoFlag) {
            autoFlag.value = "0";
          }
        }
        refreshStepEstimate(form);
      }
    });
    form.addEventListener("change", (event) => {
      if (event.target.matches("[name='dataset_id'], [name='dataset_version_id'], [name='preset_id'], [name='recipe_v2_id'], [name='cache_latents'], [name='cache_text_encoder_outputs'], [name='network_train_unet_only']")) {
        const targetInput = form.querySelector("[data-step-target]");
        if (targetInput && event.target.matches("[name='preset_id'], [name='recipe_v2_id']")) {
          targetInput.dataset.manualTarget = "false";
        }
        refreshStepEstimate(form);
      }
    });
    form.querySelector("[data-step-suggest-button]")?.addEventListener("click", () => refreshStepEstimate(form, true));
    form.querySelector("[data-step-target]")?.addEventListener("input", (event) => {
      event.target.dataset.manualTarget = "true";
    });
    refreshStepEstimate(form);
  });
}

function recipeLabel(recipe) {
  if (!recipe) {
    return uiText("recipeNotSelected");
  }
  return recipe.localized_label || recipe.short_label || recipe.display_name || recipe.name || recipe.id;
}

function recipeFullLabel(recipe) {
  if (!recipe) {
    return uiText("recipeNotSelected");
  }
  return recipe.localized_full_label || recipe.full_label || recipe.display_name || recipe.name || recipe.id;
}

function currentModeLabel(form) {
  const mode = form.querySelector("[data-recipe-mode-input]")?.value || "purpose";
  const modeLabels = {
    purpose: uiText("chooseByPurpose"),
    optimizer: uiText("chooseByOptimizer"),
    derived: uiText("deriveFromJob"),
    custom: uiText("fullCustom"),
  };
  return modeLabels[mode] || mode;
}

function currentRecipe(form) {
  const recipes = parseJsonScript("[data-recipes-v2-json]", form);
  const select = form.querySelector("[data-recipe-select]");
  return recipes.find((item) => item.id === select?.value) || null;
}

function recipeKeyParams(recipe) {
  const params = recipe?.basic_params || recipe?.params || {};
  const parts = [];
  if (params.repeats !== undefined) parts.push(`repeats ${params.repeats}`);
  if (params.max_train_epochs !== undefined) parts.push(`epoch ${params.max_train_epochs}`);
  if (params.train_batch_size !== undefined) parts.push(`batch ${params.train_batch_size}`);
  if (params.network_dim !== undefined) parts.push(`dim ${params.network_dim}`);
  return parts.join(" / ") || "-";
}

function paramsEqual(a, b) {
  return JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
}

function formatParamValue(value) {
  if (value === undefined) return "-";
  if (value === null) return "null";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function resolveWizardParams(form) {
  return stepEstimatorParams(form);
}

function setListItems(list, items) {
  if (!list) return;
  list.innerHTML = "";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    list.appendChild(li);
  });
}

function wizardCompatibility(form, recipe, params, rawText = "") {
  const errors = [];
  const warnings = [];
  const notes = [];
  const mode = form.querySelector("[data-recipe-mode-input]")?.value || "purpose";
  if (!recipe) {
    if (mode === "custom") {
      notes.push(uiText("recipeMissingCustom"));
    } else {
      errors.push(uiText("recipeMissingError"));
    }
  } else {
    notes.push(`Recipe: ${recipeLabel(recipe)}`);
    notes.push(`Target steps: ${recipe.target_steps_min ?? "-"} / ${recipe.target_steps_recommended ?? "-"} / ${recipe.target_steps_max ?? "-"}`);
    if (recipe.optimizer_lr_semantics && recipe.optimizer_lr_semantics !== "normal_lr") {
      notes.push(`LR意味: ${recipe.optimizer_lr_semantics}`);
    }
    if (recipe.risk_note) {
      notes.push(recipe.risk_note);
    }
    const validationStatus = recipe.optimizer_profile_validation_status || "untested";
    const miniPilotStatus = recipe.optimizer_profile_mini_pilot_status || "untested";
    if (validationStatus === "untested") {
      warnings.push(uiText("smokeUntested"));
    } else if (validationStatus === "prepare_ok") {
      warnings.push(uiText("prepareOnly"));
    } else if (validationStatus === "smoke_failed") {
      warnings.push(uiText("smokeFailed"));
    } else if (validationStatus === "disabled") {
      warnings.push(uiText("disabledProfile"));
    } else if (validationStatus === "smoke_ok" || validationStatus === "mini_pilot_ok") {
      notes.push(`Optimizer Profile validation: ${validationStatus}`);
    }
    if (miniPilotStatus === "untested") {
      warnings.push(uiText("miniUntested"));
    } else if (miniPilotStatus === "mini_pilot_failed") {
      warnings.push(uiText("miniFailed"));
    } else if (miniPilotStatus === "mini_pilot_warning") {
      warnings.push(uiText("miniWarning"));
    } else if (miniPilotStatus === "mini_pilot_ok") {
      notes.push(uiText("miniOk"));
    }
    const category = String(recipe.optimizer_category || "");
    if (category.includes("advanced") || category.includes("experimental")) {
      warnings.push(`${category} ${uiText("advancedOptimizer")}`);
    }
    if (recipe.network_type_availability && recipe.network_type_availability !== "available") {
      errors.push(uiFormat("networkUnavailable", { value: recipe.network_type_availability }));
    }
  }
  const te1 = Number(params.text_encoder_lr1 ?? params.text_encoder_lr ?? 0) || 0;
  const te2 = Number(params.text_encoder_lr2 ?? 0) || 0;
  if (params.cache_text_encoder_outputs && (te1 > 0 || te2 > 0)) {
    errors.push(uiText("cacheTeError"));
  }
  if (params.network_train_unet_only && (te1 > 0 || te2 > 0)) {
    errors.push(uiText("unetTeError"));
  }
  const optimizer = String(params.optimizer_type || recipe?.optimizer_definition_id || "");
  const scheduler = String(params.lr_scheduler || "");
  if ((optimizer.includes("DAdapt") || optimizer === "Prodigy") && scheduler && scheduler !== "constant") {
    warnings.push(uiText("schedulerWarning"));
  }
  if (optimizer === "Adafactor" || recipe?.optimizer_lr_semantics === "relative_step") {
    notes.push(uiText("adafactorNote"));
  }
  if (rawText.trim()) {
    try {
      const raw = JSON.parse(rawText);
      if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
        errors.push(uiText("rawArgsObject"));
      } else {
        warnings.push(uiText("rawArgsWarning"));
      }
    } catch (error) {
      errors.push(uiFormat("rawArgsParse", { message: error.message }));
    }
  }
  return { errors, warnings, notes };
}

function updateRecipeDetail(form) {
  const detail = form.querySelector("[data-recipe-detail]");
  const recipe = currentRecipe(form);
  if (!detail) return;
  if (!recipe) {
    const mode = form.querySelector("[data-recipe-mode-input]")?.value || "purpose";
    const visibleCards = Array.from(form.querySelectorAll("[data-recipe-card]")).filter((card) => !card.hidden).length;
    const message = visibleCards
      ? uiText("filterHint")
      : uiText("noFilterRecipes");
    const customNote = mode === "custom" ? uiText("customNoRecipe") : message;
    detail.innerHTML = `<strong>${uiText("recipeNotSelected")}</strong><p class="muted">${customNote}</p>`;
    return;
  }
  detail.innerHTML = `
    <strong>${recipeLabel(recipe)}</strong>
    <dl>
      <dt>${uiText("purpose")}</dt><dd>${recipe.purpose_display_name || recipe.training_purpose_id || "-"}</dd>
      <dt>Optimizer</dt><dd>${recipe.optimizer_display_name || recipe.optimizer_definition_id || "-"} / ${recipe.optimizer_category || "-"}</dd>
      <dt>Profile</dt><dd>${recipe.optimizer_profile_display_name || recipe.optimizer_profile_id || "-"}</dd>
      <dt>Network</dt><dd>${recipe.network_type_display_name || recipe.network_type_id || "-"} / ${recipe.network_type_availability || "-"}</dd>
      <dt>Target steps</dt><dd>${recipe.target_steps_min ?? "-"} / <strong>${recipe.target_steps_recommended ?? "-"}</strong> / ${recipe.target_steps_max ?? "-"}</dd>
      <dt>${uiText("expectedBehavior")}</dt><dd>${recipe.expected_behavior || "-"}</dd>
      <dt>${uiText("note")}</dt><dd>${recipe.risk_note || recipe.optimizer_risk_note || "-"}</dd>
    </dl>
    <p><a href="/training-recipes/${encodeURIComponent(recipe.id)}">${uiText("details")}</a> / <a href="/optimizers/${encodeURIComponent(recipe.optimizer_definition_id || "")}">Optimizer ${uiText("details")}</a></p>
  `;
}

function updateSelectedRecipePanel(form) {
  const panel = form.querySelector("[data-selected-recipe-panel]");
  const summary = form.querySelector("[data-selected-recipe-summary]");
  if (!panel || !summary) return;
  const recipe = currentRecipe(form);
  const params = resolveWizardParams(form);
  const result = wizardCompatibility(form, recipe, params, form.querySelector("[data-param-editor-raw]")?.value || "");
  const overrideCount = updateParamDiff(form);
  const stepTotal = form.querySelector("[data-step-field='total_steps']")?.textContent || "-";
  const createButton = form.querySelector("[data-selected-create-button]");
  if (!recipe) {
    summary.innerHTML = `<strong>${uiText("recipeNotSelected")}</strong><p class="muted">${uiText("selectedHint")}</p>`;
    if (createButton) createButton.disabled = true;
    return;
  }
  const riskClass = String(recipe.optimizer_category || "").includes("experimental") ? "warning" : "";
  const validation = recipeValidationBadge(recipe);
  summary.innerHTML = `
    <strong>${recipeLabel(recipe)}</strong>
    <dl>
      <dt>Optimizer</dt><dd>${recipe.optimizer_display_name || recipe.optimizer_definition_id || "-"}</dd>
      <dt>Purpose</dt><dd>${recipe.purpose_display_name || recipe.training_purpose_id || "-"}</dd>
      <dt>Target steps</dt><dd>${recipe.target_steps_min ?? "-"} / <strong>${recipe.target_steps_recommended ?? "-"}</strong> / ${recipe.target_steps_max ?? "-"}</dd>
      <dt>Expected steps</dt><dd>${stepTotal}</dd>
      <dt>Compatibility</dt><dd>${result.errors.length ? `ERROR ${result.errors.length}` : (result.warnings.length ? `WARNING ${result.warnings.length}` : "OK")}</dd>
      <dt>Smoke</dt><dd><span class="label ${validation.klass}">${validation.text}</span></dd>
      <dt>Override</dt><dd>${overrideCount} ${uiText("countSuffix")}</dd>
      <dt>${uiText("risk")}</dt><dd><span class="label ${riskClass}">${recipe.risk_note || recipe.optimizer_category || "-"}</span></dd>
    </dl>
  `;
  if (createButton) {
    createButton.disabled = result.errors.length > 0;
  }
}

function updateParamDiff(form) {
  const recipe = currentRecipe(form);
  const mode = form.querySelector("[data-recipe-mode-input]")?.value || "purpose";
  const base = (recipe && recipe.params) || {};
  const resolved = resolveWizardParams(form);
  const preview = form.querySelector("[data-resolved-params-preview]");
  if (preview) {
    if (!recipe && mode !== "custom") {
      preview.textContent = uiText("noRecipeParams");
    } else {
      preview.textContent = JSON.stringify(resolved, null, 2);
    }
  }
  const changed = Object.keys(resolved)
    .filter((key) => !paramsEqual(base[key], resolved[key]))
    .sort();
  const diff = form.querySelector("[data-param-diff-preview]");
  if (diff) {
    if (!changed.length) {
      diff.textContent = uiText("noDiff");
    } else {
      const rows = changed.map((key) => `<tr><td><code>${key}</code></td><td>${formatParamValue(base[key])}</td><td>${formatParamValue(resolved[key])}</td></tr>`).join("");
      diff.innerHTML = `<table class="compact-table"><thead><tr><th>key</th><th>Recipe</th><th>${uiText("recipeValue")}</th></tr></thead><tbody>${rows}</tbody></table>`;
    }
  }
  const summaryDiff = form.querySelector("[data-summary-diff]");
  if (summaryDiff) summaryDiff.textContent = `${changed.length} ${uiText("countSuffix")}`;
  return changed.length;
}

function updateCompatibilityPanel(form) {
  const recipe = currentRecipe(form);
  const params = resolveWizardParams(form);
  const rawText = form.querySelector("[data-param-editor-raw]")?.value || "";
  const result = wizardCompatibility(form, recipe, params, rawText);
  const panel = form.querySelector("[data-compatibility-panel]");
  if (!panel) return result;
  panel.classList.toggle("warning", result.errors.length > 0 || result.warnings.length > 0);
  const errorCount = panel.querySelector("[data-compat-error-count]");
  const warningCount = panel.querySelector("[data-compat-warning-count]");
  const noteCount = panel.querySelector("[data-compat-note-count]");
  if (errorCount) errorCount.textContent = result.errors.length;
  if (warningCount) warningCount.textContent = result.warnings.length;
  if (noteCount) noteCount.textContent = result.notes.length;
  setListItems(panel.querySelector("[data-compat-errors]"), result.errors.length ? result.errors : [uiText("noErrors")]);
  setListItems(panel.querySelector("[data-compat-warnings]"), result.warnings.length ? result.warnings : [uiText("noWarnings")]);
  setListItems(panel.querySelector("[data-compat-notes]"), result.notes);
  const submit = form.querySelector("[data-wizard-submit]");
  if (submit) {
    submit.disabled = result.errors.length > 0;
  }
  const compat = form.querySelector("[data-summary-compat]");
  if (compat) {
    compat.textContent = result.errors.length ? `ERROR ${result.errors.length}` : (result.warnings.length ? `WARNING ${result.warnings.length}` : "OK");
  }
  return result;
}

function updateWizardSummary(form) {
  const mode = form.querySelector("[data-recipe-mode-input]")?.value || "purpose";
  const modeLabels = {
    purpose: uiText("chooseByPurpose"),
    optimizer: uiText("chooseByOptimizer"),
    derived: uiText("deriveFromJob"),
    custom: uiText("fullCustom"),
  };
  const modeNode = form.querySelector("[data-summary-mode]");
  if (modeNode) modeNode.textContent = modeLabels[mode] || mode;
  const recipeNode = form.querySelector("[data-summary-recipe]");
  if (recipeNode) recipeNode.textContent = recipeLabel(currentRecipe(form));
}

function updateWizardState(form) {
  updateRecipeDetail(form);
  updateParamDiff(form);
  updateCompatibilityPanel(form);
  updateWizardSummary(form);
  updateSelectedRecipePanel(form);
}

function recipeMatchesFilters(recipe, filters) {
  return Object.entries(filters).every(([key, value]) => {
    return !value || String(recipe[key] || "") === String(value);
  });
}

function recipeOptimizerGroup(recipe) {
  const category = String(recipe.optimizer_category || "");
  if (category.includes("experimental")) return "Experimental";
  if (category.includes("auto_lr") || category.includes("advanced")) return "Auto-LR / Advanced";
  if (category.includes("memory")) return "Memory Saving";
  if (category.includes("stable")) return "Stable";
  return "Custom";
}

function recipePurposeGroup(recipe) {
  return recipe.purpose_display_name || recipe.training_purpose_id || "Custom";
}

function recipeValidationBadge(recipe) {
  const miniStatus = recipe?.optimizer_profile_mini_pilot_status || "untested";
  const status = miniStatus !== "untested" ? miniStatus : (recipe?.optimizer_profile_validation_status || "untested");
  const map = {
    untested: { text: "Untested", klass: "warning" },
    prepare_ok: { text: "Prepare OK", klass: "ok" },
    smoke_ok: { text: "Smoke OK", klass: "ok" },
    smoke_failed: { text: "Failed", klass: "error" },
    mini_pilot_ok: { text: "Mini Pilot OK", klass: "ok" },
    mini_pilot_warning: { text: "Mini Pilot Warning", klass: "warning" },
    mini_pilot_failed: { text: "Mini Pilot Failed", klass: "error" },
    disabled: { text: "Disabled", klass: "error" },
  };
  return map[status] || { text: status, klass: "warning" };
}

function appendRecipeCard(form, grid, recipe) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "recipe-choice-card";
    button.setAttribute("data-recipe-card", recipe.id);
    button.setAttribute("data-model-family", recipe.model_family || "");
    button.setAttribute("data-purpose-id", recipe.training_purpose_id || "");
    button.setAttribute("data-optimizer-id", recipe.optimizer_definition_id || "");
    button.setAttribute("data-profile-id", recipe.optimizer_profile_id || "");
    button.setAttribute("data-network-id", recipe.network_type_id || "");
    button.setAttribute("data-recipe-type", recipe.recipe_type || "");
    const risk = recipe.risk_note || recipe.optimizer_risk_note || "";
    const recommended = recipe.recommended_badge || ((recipe.recipe_type === "balanced" || recipe.recipe_type === "balanced_long") ? uiText("recommended") : "");
    const difficulty = recipe.difficulty_label || recipe.optimizer_category || "-";
    const validation = recipeValidationBadge(recipe);
    button.innerHTML = `
      ${recommended ? `<span class="recipe-recommend-badge">${recommended}</span>` : ''}
      <strong>${recipeLabel(recipe)}</strong>
      <span>${recipe.card_subtitle || `${recipe.optimizer_display_name || recipe.optimizer_definition_id || "-"} / ${recipe.recipe_type || "-"}`}</span>
      <small>${recipe.model_family || "-"} / ${recipe.purpose_display_name || recipe.training_purpose_id || "-"} / ${recipe.optimizer_display_name || recipe.optimizer_definition_id || "-"}</small>
      <small><span class="label ${validation.klass}">${validation.text}</span> ${difficulty} / target ${recipe.target_steps_min ?? "-"} / ${recipe.target_steps_recommended ?? "-"} / ${recipe.target_steps_max ?? "-"}</small>
      <small>key params: ${recipeKeyParams(recipe)}</small>
      <small>${recipe.expected_behavior || "-"}</small>
      ${risk ? `<span class="label warning">${risk}</span>` : ''}
      <span class="button small-button recipe-card-action">${uiText("chooseRecipe")}</span>
    `;
    button.addEventListener("click", () => {
      const select = form.querySelector("[data-recipe-select]");
      if (select) {
        select.value = recipe.id;
        select.dispatchEvent(new Event("change", { bubbles: true }));
      }
    });
    grid.appendChild(button);
}

function renderRecipeCards(form, filters = {}) {
  const grid = form.querySelector("[data-recipe-cards]");
  if (!grid) return;
  const recipes = parseJsonScript("[data-recipes-v2-json]", form);
  const mode = form.querySelector("[data-recipe-mode-input]")?.value || "purpose";
  const filtered = recipes.filter((recipe) => recipeMatchesFilters(recipe, filters));
  const selectedId = form.querySelector("[data-recipe-select]")?.value || "";
  const countNode = form.querySelector("[data-recipe-result-count]");
  const hintNode = form.querySelector("[data-recipe-group-hint]");
  const emptyTemplate = form.querySelector("[data-recipe-empty-state]");
  if (countNode) countNode.textContent = uiFormat("recipeCount", { count: filtered.length });
  if (hintNode) {
    hintNode.textContent = mode === "optimizer"
      ? uiText("optimizerGroupingHint")
      : uiText("purposeGroupingHint");
  }
  grid.innerHTML = "";

  if (!filtered.length) {
    if (emptyTemplate) emptyTemplate.hidden = false;
    return;
  }
  if (emptyTemplate) emptyTemplate.hidden = true;

  const groups = new Map();
  filtered.forEach((recipe) => {
    const groupName = mode === "optimizer" ? recipePurposeGroup(recipe) : recipeOptimizerGroup(recipe);
    if (!groups.has(groupName)) groups.set(groupName, []);
    groups.get(groupName).push(recipe);
  });

  const order = mode === "optimizer"
    ? ["Character Face", "Style", "Costume", "Custom"]
    : ["Stable", "Memory Saving", "Auto-LR / Advanced", "Experimental", "Custom"];
  Array.from(groups.keys())
    .sort((a, b) => {
      const ai = order.indexOf(a);
      const bi = order.indexOf(b);
      if (ai !== -1 || bi !== -1) return (ai === -1 ? 999 : ai) - (bi === -1 ? 999 : bi);
      return a.localeCompare(b);
    })
    .forEach((groupName) => {
      const heading = document.createElement("h3");
      heading.className = "recipe-card-group";
      heading.textContent = groupName;
      grid.appendChild(heading);
      groups.get(groupName)
        .sort((a, b) => Number(a.sort_order || 9999) - Number(b.sort_order || 9999) || recipeFullLabel(a).localeCompare(recipeFullLabel(b)))
        .forEach((recipe) => {
          appendRecipeCard(form, grid, recipe);
          const card = grid.querySelector(`[data-recipe-card="${CSS.escape(recipe.id)}"]`);
          if (card) card.classList.toggle("active", card.getAttribute("data-recipe-card") === selectedId);
        });
    });
}

function optionBaseLabel(option) {
  if (!option.dataset.baseLabel) {
    option.dataset.baseLabel = option.textContent.replace(/\s+\(\d+\)$/, "");
  }
  return option.dataset.baseLabel;
}

function updateFilterOptionCounts(form, currentFilters) {
  const recipes = parseJsonScript("[data-recipes-v2-json]", form);
  form.querySelectorAll("[data-recipe-filter]").forEach((select) => {
    const key = select.getAttribute("data-recipe-filter");
    if (!key) return;
    Array.from(select.options).forEach((option) => {
      const base = optionBaseLabel(option);
      if (!option.value) {
        option.textContent = base;
        option.disabled = false;
        option.hidden = false;
        return;
      }
      const testFilters = { ...currentFilters, [key]: option.value };
      const count = recipes.filter((recipe) => recipeMatchesFilters(recipe, testFilters)).length;
      option.textContent = `${base} (${count})`;
      option.disabled = count === 0;
      option.hidden = count === 0;
    });
  });
}

function updateModeSummary(form, { expanded = false } = {}) {
  const label = form.querySelector("[data-mode-summary-label]");
  const cards = form.querySelector("[data-mode-cards]");
  if (label) label.textContent = currentModeLabel(form);
  if (cards) {
    if (arguments.length > 1) {
      cards.dataset.expanded = expanded ? "true" : "false";
    }
    cards.classList.toggle("collapsed", cards.dataset.expanded !== "true");
  }
}

function buildRecipeCards(form) {
  renderRecipeCards(form, {});
}

function syncOptimizerProfileFilter(form) {
  const optimizerSelect = form.querySelector("[data-recipe-filter='optimizer_definition_id']");
  const profileSelect = form.querySelector("[data-recipe-filter='optimizer_profile_id']");
  if (!optimizerSelect || !profileSelect) {
    return;
  }
  const recipeSelect = form.querySelector("[data-recipe-select]");
  const optimizerId = optimizerSelect.value || "";
  const modelFamily = form.querySelector("[data-recipe-filter='model_family']")?.value || "";
  const recipeType = form.querySelector("[data-recipe-filter='recipe_type']")?.value || "";
  const networkType = form.querySelector("[data-recipe-filter='network_type_id']")?.value || "";
  const purposeId = form.querySelector("[data-recipe-filter='training_purpose_id']")?.value || "";
  const recipeOptions = recipeSelect ? Array.from(recipeSelect.options).filter((option) => option.value) : [];
  let selectedVisible = false;
  Array.from(profileSelect.options).forEach((option) => {
    const profileOptimizerId = option.getAttribute("data-optimizer-id") || "";
    const hasMatchingRecipe = !option.value || recipeOptions.some((recipeOption) => {
      return recipeOption.getAttribute("data-profile-id") === option.value
        && (!optimizerId || recipeOption.getAttribute("data-optimizer-id") === optimizerId)
        && (!modelFamily || recipeOption.getAttribute("data-model-family") === modelFamily)
        && (!recipeType || recipeOption.getAttribute("data-recipe-type") === recipeType)
        && (!networkType || recipeOption.getAttribute("data-network-id") === networkType)
        && (!purposeId || recipeOption.getAttribute("data-purpose-id") === purposeId);
    });
    const visible = !option.value || ((!optimizerId || profileOptimizerId === optimizerId) && hasMatchingRecipe);
    option.hidden = !visible;
    option.disabled = !visible;
    if (visible && option.selected) {
      selectedVisible = true;
    }
  });
  if (!selectedVisible) {
    profileSelect.value = "";
  }
}

function updateOptimizerInfoPanel(form) {
  const panel = form.querySelector("[data-optimizer-info-panel]");
  if (!panel) return;
  const mode = form.querySelector("[data-recipe-mode-input]")?.value || "purpose";
  const optimizerId = form.querySelector("[data-recipe-filter='optimizer_definition_id']")?.value || "";
  if (mode !== "optimizer" || !optimizerId) {
    panel.hidden = true;
    panel.innerHTML = "";
    return;
  }
  const optimizers = parseJsonScript("[data-optimizers-v2-json]", form);
  const optimizer = optimizers.find((item) => item.id === optimizerId);
  if (!optimizer) {
    panel.hidden = true;
    panel.innerHTML = "";
    return;
  }
  const notes = Array.isArray(optimizer.compatibility_notes) ? optimizer.compatibility_notes.join(" / ") : "";
  panel.hidden = false;
  panel.innerHTML = `
    <strong>${optimizer.display_name || optimizer.id}</strong>
    <dl>
      <dt>LR意味</dt><dd>${optimizer.lr_semantics || "-"}</dd>
      <dt>default LR</dt><dd>${optimizer.default_learning_rate ?? "-"} / UNet ${optimizer.default_unet_lr ?? "-"} / TE ${optimizer.default_text_encoder_lr ?? "-"}</dd>
      <dt>target steps</dt><dd>${optimizer.target_steps_min ?? "-"} / <strong>${optimizer.target_steps_recommended ?? "-"}</strong> / ${optimizer.target_steps_max ?? "-"}</dd>
      <dt>${uiText("risk")}</dt><dd>${optimizer.risk_note || "-"}</dd>
      <dt>${uiText("compatibilityNotes")}</dt><dd>${notes || "-"}</dd>
    </dl>
  `;
}

function applyRecipeFilters(form) {
  const mode = form.querySelector("[data-recipe-mode-input]")?.value || "purpose";
  const recipeSelect = form.querySelector("[data-recipe-select]");
  if (!recipeSelect) {
    return;
  }
  syncOptimizerProfileFilter(form);
  const purposeField = form.querySelector("[data-purpose-filter-field]");
  const optimizerField = form.querySelector("[data-optimizer-filter-field]");
  const profileField = form.querySelector("[data-profile-filter-field]");
  const networkField = form.querySelector("[data-network-filter-field]");
  if (purposeField) purposeField.hidden = mode === "optimizer";
  if (optimizerField) optimizerField.hidden = mode === "purpose";
  if (profileField) profileField.hidden = mode !== "optimizer";
  if (networkField) networkField.hidden = mode !== "custom";
  const parentField = form.querySelector("[data-derived-parent-field]");
  if (parentField) parentField.hidden = mode !== "derived";

  const filters = {};
  form.querySelectorAll("[data-recipe-filter]").forEach((filter) => {
    const key = filter.getAttribute("data-recipe-filter");
    if (key && filter.value) {
      filters[key] = filter.value;
    }
  });
  if (mode === "purpose") {
    delete filters.optimizer_definition_id;
    delete filters.optimizer_profile_id;
    delete filters.network_type_id;
  }
  if (mode === "optimizer") {
    delete filters.training_purpose_id;
    delete filters.network_type_id;
  }
  if (mode !== "custom") {
    delete filters.network_type_id;
  }
  updateFilterOptionCounts(form, filters);

  let selectedVisible = false;
  Array.from(recipeSelect.options).forEach((option) => {
    if (!option.value) {
      option.hidden = false;
      option.disabled = false;
      return;
    }
    const visible = Object.entries(filters).every(([key, value]) => {
      const attr = {
        model_family: "model-family",
        training_purpose_id: "purpose-id",
        optimizer_definition_id: "optimizer-id",
        optimizer_profile_id: "profile-id",
        network_type_id: "network-id",
        recipe_type: "recipe-type",
      }[key];
      return !value || option.getAttribute(`data-${attr}`) === value;
    });
    option.hidden = !visible;
    option.disabled = !visible;
    if (visible && option.selected) {
      selectedVisible = true;
    }
  });
  if (!selectedVisible && recipeSelect.value) {
    recipeSelect.value = "";
  }
  renderRecipeCards(form, filters);
  updateOptimizerInfoPanel(form);
  updateModeSummary(form);
  updateWizardState(form);
}

function updateParentJobSummary(form) {
  const select = form.querySelector("select[name='parent_job_id']");
  const summary = form.querySelector("[data-parent-job-summary]");
  if (!select || !summary) {
    return;
  }
  const option = select.options[select.selectedIndex];
  const hasParent = Boolean(option && option.value);
  summary.hidden = !hasParent;
  if (!hasParent) {
    return;
  }
  const setField = (key, value) => {
    const node = summary.querySelector(`[data-parent-summary-field='${key}']`);
    if (node) {
      node.textContent = value || "-";
    }
  };
  setField("project", option.getAttribute("data-parent-project"));
  setField("status", option.getAttribute("data-parent-status"));
  setField("recipe", option.getAttribute("data-parent-recipe"));
  setField("optimizer", option.getAttribute("data-parent-optimizer"));
  setField("purpose", option.getAttribute("data-parent-purpose"));
  setField("steps", `${option.getAttribute("data-parent-steps") || "-"} / ${option.getAttribute("data-parent-step-status") || "-"}`);
  setField("params", `repeats ${option.getAttribute("data-parent-repeats") || "-"} / epochs ${option.getAttribute("data-parent-epochs") || "-"} / batch ${option.getAttribute("data-parent-batch") || "-"}`);
}

function initRecipeSelectors() {
  document.querySelectorAll("form").forEach((form) => {
    if (!form.querySelector("[data-recipe-select]")) {
      return;
    }
    buildRecipeCards(form);
    form.querySelectorAll("[data-mode-card]").forEach((button) => {
      button.addEventListener("click", () => {
        const mode = button.getAttribute("data-mode-card") || "purpose";
        const input = form.querySelector("[data-recipe-mode-input]");
        if (input) input.value = mode;
        form.querySelectorAll("[data-mode-card]").forEach((card) => card.classList.toggle("active", card === button));
        if (mode === "custom") {
          const select = form.querySelector("[data-recipe-select]");
          if (select) {
            select.value = "";
            select.dispatchEvent(new Event("change", { bubbles: true }));
          }
        }
        updateModeSummary(form, { expanded: false });
        applyRecipeFilters(form);
        refreshStepEstimate(form);
      });
    });
    form.querySelector("[data-mode-change-button]")?.addEventListener("click", () => {
      updateModeSummary(form, { expanded: true });
    });
    const currentMode = form.querySelector("[data-recipe-mode-input]")?.value || "purpose";
    form.querySelector(`[data-mode-card='${currentMode}']`)?.classList.add("active");
    updateModeSummary(form, { expanded: !new URLSearchParams(window.location.search).has("mode") });
    applyRecipeFilters(form);
    updateWizardState(form);
    form.addEventListener("change", (event) => {
      if (event.target.matches("[data-recipe-filter]")) {
        applyRecipeFilters(form);
        refreshStepEstimate(form);
      }
      if (event.target.matches("[data-recipe-select]")) {
        const selectedId = event.target.value;
        form.querySelectorAll("[data-recipe-card]").forEach((card) => card.classList.toggle("active", card.getAttribute("data-recipe-card") === selectedId));
        updateWizardState(form);
        refreshStepEstimate(form);
      }
      if (event.target.matches("[data-param-editor-key], [data-param-editor-raw]")) {
        updateWizardState(form);
      }
      if (event.target.matches("select[name='parent_job_id']")) {
        updateParentJobSummary(form);
      }
    });
    form.addEventListener("input", (event) => {
      if (event.target.matches("[data-param-editor-key], [data-param-editor-raw], [data-param-editor-reason]")) {
        updateWizardState(form);
      }
    });
    form.addEventListener("submit", (event) => {
      const result = updateCompatibilityPanel(form);
      if (result.errors.length) {
        event.preventDefault();
        showPageNotice("Compatibility CheckにERRORがあります。修正してから作成してください。", "warning", form);
      }
    });
    updateParentJobSummary(form);
  });
}

function updateReviewAutomationDefaults(container, { initial = false } = {}) {
  if (!container) {
    return;
  }
  const mode = container.querySelector("[data-review-automation-mode]");
  const images = container.querySelector("[data-review-automation-images]");
  const runtime = container.querySelector("[data-review-automation-runtime]");
  const note = container.querySelector("[data-review-automation-note]");
  if (!mode || !images || !runtime) {
    return;
  }

  const currentImages = String(images.value || "").trim();
  const currentRuntime = String(runtime.value || "").trim();
  const isLegacyQuickImages = currentImages === "" || currentImages === "18";
  const isLegacyQuickRuntime = currentRuntime === "" || currentRuntime === "20" || currentRuntime === "60";
  const isStandardImages = currentImages === "150";
  const isStandardRuntime = currentRuntime === "240";

  if (mode.value === "standard_auto") {
    if (!initial || isLegacyQuickImages) {
      images.value = "150";
    }
    if (!initial || isLegacyQuickRuntime) {
      runtime.value = "240";
    }
    if (note) {
      note.textContent = uiText("reviewStandardAutoNote");
    }
    return;
  }

  if (mode.value === "quick_auto") {
    if (!initial || isStandardImages) {
      images.value = "18";
    }
    if (!initial || isStandardRuntime) {
      runtime.value = "60";
    }
    if (note) {
      note.textContent = uiText("reviewQuickAutoNote");
    }
    return;
  }

  if (note) {
    note.textContent = mode.value === "manual"
      ? uiText("reviewManualNote")
      : uiText("reviewPlanOnlyNote");
  }
}

function initReviewAutomationSettings() {
  document.querySelectorAll("[data-review-automation-settings]").forEach((container) => {
    updateReviewAutomationDefaults(container, { initial: true });
    container.querySelector("[data-review-automation-mode]")?.addEventListener("change", () => {
      updateReviewAutomationDefaults(container);
    });
  });
}

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
    button.dataset.reviewPreparationBusy = "1";
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
      delete button.dataset.reviewPreparationBusy;
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
      const coverageUpdated = updateEmbeddingCoverage(payload.coverage);
      if (payload.status === "completed") {
        statusText.textContent = `完了: ${payload.ready_count ?? processed} / ${total} 件`;
        panel.classList.remove("warning");
        panel.classList.add("success");
        enableEmbeddingButtons();
        window.setTimeout(async () => {
          const updated = await refreshMachineReviewReadiness();
          if (!updated && !coverageUpdated) {
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
  schedulePageRefresh({
    target,
    paramsToDelete: ["embedding_message", "embedding_job_id", "embedding_error"],
    delayMs: 0,
    restoreScroll: true,
  });
}

function schedulePageRefresh({ target = window.location.href, paramsToDelete = [], hash = "", delayMs = 900, restoreScroll = false } = {}) {
  if (document.body.hasAttribute("data-page-refresh-scheduled")) {
    return;
  }
  document.body.setAttribute("data-page-refresh-scheduled", "1");
  const url = new URL(target, window.location.href);
  paramsToDelete.forEach((key) => url.searchParams.delete(key));
  if (hash) {
    url.hash = hash;
  }
  if (!url.hash) {
    url.hash = window.location.hash || "";
  }
  if (restoreScroll && !url.hash) {
    sessionStorage.setItem("loraStudioRestoreScrollY", String(window.scrollY || 0));
  }
  window.setTimeout(() => {
    window.location.href = url.toString();
    window.location.reload();
  }, delayMs);
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

function updateEmbeddingCoverage(coverage) {
  if (!coverage) {
    return false;
  }
  let updated = false;
  Object.entries(coverage).forEach(([key, value]) => {
    document.querySelectorAll(`[data-embedding-coverage-field="${key}"], [data-reference-embedding-field="${key}"]`).forEach((target) => {
      target.textContent = value ?? 0;
      updated = true;
    });
  });
  return updated;
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
  const terminalStatuses = ["completed", "failed", "stopped"];
  const runButton = section.querySelector("[data-review-preparation-run-button]");
  if (runButton && ["completed", "failed", "stopped"].includes(payload.status)) {
    if (payload.status === "completed") {
      runButton.disabled = true;
      runButton.textContent = payload.matrix_ready ? "Matrix作成済み" : "完了";
    } else {
      runButton.disabled = false;
      runButton.textContent = runButton.dataset.originalText || "候補レビューを開始";
    }
  }
  if (terminalStatuses.includes(payload.status)) {
    document.querySelectorAll("[data-review-preparation-run-button][data-review-preparation-busy='1']").forEach((button) => {
      delete button.dataset.reviewPreparationBusy;
      if (payload.status === "completed" && payload.matrix_ready && payload.matrix_url) {
        const link = document.createElement("a");
        link.className = "button";
        link.href = payload.matrix_url;
        link.textContent = "レビューMatrixを開く";
        const form = button.closest("form");
        if (form) {
          form.replaceWith(link);
        } else {
          button.replaceWith(link);
        }
        return;
      }
      button.disabled = false;
      button.textContent = button.dataset.originalText || (payload.status === "completed" ? "レビュー準備完了" : "候補レビューを開始");
    });
    const message = payload.status === "completed"
      ? (payload.matrix_ready ? "レビュー準備は完了しています。レビューMatrixを開けます。" : "レビュー準備は完了しています。レビューMatrixを作成できます。")
      : "レビュー準備は終了しています。ログを確認してください。";
    showPageNotice(message, payload.status === "completed" ? "info" : "warning", anchorForm);
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
  const runningStatuses = ["starting", "running", "generating_images", "embedding_images", "machine_reviewing", "building_matrix"];
  if (sessionId && runningStatuses.includes(status)) {
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

function clearTransientNoticeParams() {
  clearQueryNoticeParams([
    "embedding_error",
    "embedding_message",
    "embedding_job_id",
    "machine_review_error",
    "machine_review_message",
    "machine_review_job_id",
    "generation_error",
    "generation_message",
    "review_prepare",
    "review_prepare_error",
  ]);
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
        schedulePageRefresh({
          paramsToDelete: ["machine_review_message", "machine_review_job_id", "machine_review_error"],
        });
        return true;
      }
      if (payload.status === "failed") {
        statusText.textContent = `失敗: ${payload.error_message || ""}`;
        panel.classList.add("warning");
        clearQueryNoticeParams(["machine_review_message", "machine_review_job_id"]);
        return true;
      }
      if (payload.status === "stopped") {
        statusText.textContent = `停止: ${processed} / ${total} 件`;
        panel.classList.add("warning");
        clearQueryNoticeParams(["machine_review_message", "machine_review_job_id"]);
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
    updateReferenceEmbeddingStatus(payload.embedding);
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
    updateReferenceEmbeddingStatus(payload.embedding);
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

function updateReferenceEmbeddingStatus(embedding) {
  if (!embedding) {
    return;
  }
  const coverage = embedding.coverage || {};
  ["total", "ready", "stale", "failed", "missing", "not_computed"].forEach((key) => {
    const target = document.querySelector(`[data-reference-embedding-field="${key}"]`);
    if (target) {
      target.textContent = coverage[key] ?? 0;
    }
  });

  const readiness = embedding.readiness || {};
  const label = document.querySelector("[data-reference-readiness-label]");
  if (label) {
    const text = readiness.label || "UNKNOWN";
    label.textContent = text;
    label.className = `label ${text.toLowerCase()}`;
  }
  const completeness = document.querySelector('[data-reference-readiness-field="completeness_label"]');
  if (completeness) {
    completeness.textContent = readiness.completeness_label || "UNKNOWN";
  }
  const imageCount = document.querySelector('[data-reference-readiness-field="image_count"]');
  if (imageCount) {
    imageCount.textContent = readiness.image_count ?? 0;
  }
  const embeddingCount = document.querySelector('[data-reference-readiness-field="embedding"]');
  if (embeddingCount) {
    embeddingCount.textContent = `${readiness.ready ?? 0} / ${readiness.total ?? 0}`;
  }
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
  const activeRows = rows.filter((row) => isActiveGenerationStatus(row.getAttribute("data-generation-status")));
  if (!activeRows.length) {
    return;
  }

  const pollRow = async (row) => {
    const runId = row.getAttribute("data-validation-run-row");
    if (!runId || !isActiveGenerationStatus(row.getAttribute("data-generation-status"))) {
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
    const activeRows = [...document.querySelectorAll("[data-validation-run-row]")]
      .filter((row) => isActiveGenerationStatus(row.getAttribute("data-generation-status")));
    if (!activeRows.length) {
      window.clearInterval(timer);
      return;
    }
    activeRows.forEach((row) => pollRow(row));
  }, 5000);
}

function isActiveGenerationStatus(status) {
  return status === "running" || status === "queued";
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
  const isActive = isActiveGenerationStatus(status);
  const runButton = container.querySelector("[data-generation-run-button]");
  const stopButton = container.querySelector("[data-generation-stop-button]");
  if (runButton) {
    runButton.disabled = isActive;
    runButton.hidden = isRunning;
    if (!isActive) {
      runButton.removeAttribute("title");
    }
  }
  if (stopButton) {
    stopButton.disabled = !isRunning;
    stopButton.hidden = !isRunning;
  }
}

function syncAllGenerationRunButtons() {
  const isAnyRunning = [...document.querySelectorAll("[data-validation-run-row]")]
    .some((row) => isActiveGenerationStatus(row.getAttribute("data-generation-status")));
  document.querySelectorAll("[data-validation-run-row]").forEach((row) => {
    const isThisRunning = isActiveGenerationStatus(row.getAttribute("data-generation-status"));
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
  const wasActive = isActiveGenerationStatus(row.getAttribute("data-generation-status"));
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
  if (wasActive && ["completed", "failed", "stopped"].includes(payload.status || "") && !document.body.hasAttribute("data-validation-generation-refreshing")) {
    document.body.setAttribute("data-validation-generation-refreshing", "1");
    showPageNotice("検証画像生成の状態が変わりました。次のRunを確認するため画面を更新します。");
    schedulePageRefresh({
      paramsToDelete: ["generation_error", "generation_message", "generation_notice"],
      hash: "#validation-runs",
      delayMs: 1200,
    });
  }
}

function initBulkValidationGenerationSubmit() {
  const form = document.querySelector("[data-bulk-generation-form]");
  if (!form) {
    return;
  }
  form.addEventListener("submit", (event) => {
    const button = form.querySelector("[data-bulk-generation-button]");
    const checked = appendSelectedValidationRunInputs(form);
    const selected = checked.length;
    if (!selected) {
      event.preventDefault();
      showPageNotice("画像生成する検証Runを選択してください。", "warning", form);
      return;
    }
    showPageNotice(`選択した検証Run ${selected} 件の画像生成を開始します。生成後に不足レビューも自動で再計算します。`, "info", form);
    window.setTimeout(() => {
      if (button) {
        button.disabled = true;
        button.textContent = "一括生成を開始中...";
      }
      document.querySelectorAll("#validation-runs button").forEach((candidate) => {
        if (candidate.type === "submit") candidate.disabled = true;
      });
    }, 0);
  });
}

function initBulkValidationAssistSubmit() {
  const form = document.querySelector("[data-bulk-assist-form]");
  if (!form) {
    return;
  }
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = form.querySelector("[data-bulk-assist-button]");
    const checked = appendSelectedValidationRunInputs(form);
    const selected = checked.length;
    if (!selected) {
      showPageNotice("Embedding計算する検証Runを選択してください。", "warning", form);
      return;
    }
    if (button) {
      button.disabled = true;
      button.dataset.originalText = button.textContent;
      button.textContent = "一括計算を開始中...";
    }
    const actionButtons = document.querySelectorAll("#validation-runs button");
    actionButtons.forEach((candidate) => {
      if (candidate.type === "submit") candidate.disabled = true;
    });
    form.dataset.bulkAssistRunning = "1";
    form.dataset.bulkAssistRunIds = checked.map((checkbox) => checkbox.value).join(",");
    showPageNotice(`選択した検証Run ${selected} 件のEmbedding / 機械補助レビューを開始します。`, "info", form);
    try {
      const response = await fetch(form.action, {
        method: "POST",
        body: new FormData(form),
        headers: { "X-Requested-With": "fetch", "Accept": "application/json" },
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok || payload.ok === false) {
        delete form.dataset.bulkAssistRunning;
        delete form.dataset.bulkAssistRunIds;
        if (payload.running_job_id) {
          actionButtons.forEach((candidate) => {
            if (candidate.type === "submit") candidate.disabled = false;
          });
          if (button) {
            button.disabled = false;
            button.textContent = button.dataset.originalText || "画像生成済みRunの不足レビューだけ再計算";
          }
          startEmbeddingJobPolling(
            payload.running_job_id,
            payload.message || `Embedding Job #${payload.running_job_id} が実行中です。`,
            window.location.href,
            form,
          );
          return;
        }
        throw new Error(payload.message || await response.text());
      }
      showPageNotice(payload.message || "Embedding / 機械補助レビューを開始しました。処理中は同じボタンを押さずに完了を待ってください。", "info", form);
      createBulkAssistStatusPanel(form, payload.message || "Embedding / 機械補助レビューを開始しました。");
      openSelectedAssistLogs();
      pollValidationAssistLogs();
    } catch (error) {
      delete form.dataset.bulkAssistRunning;
      delete form.dataset.bulkAssistRunIds;
      showPageNotice(error.message || "Embedding / 機械補助レビューを開始できませんでした。", "warning", form);
      actionButtons.forEach((candidate) => {
        if (candidate.type === "submit") candidate.disabled = false;
      });
      if (button) {
        button.disabled = false;
        button.textContent = button.dataset.originalText || "画像生成済みRunの不足レビューだけ再計算";
      }
    }
  });
}

function createBulkAssistStatusPanel(form, message) {
  const container = form.closest(".actions") || form.parentElement || document.body;
  let panel = document.querySelector("[data-bulk-assist-status]");
  if (!panel) {
    panel = document.createElement("p");
    panel.className = "notice";
    panel.setAttribute("data-bulk-assist-status", "1");
    container.before(panel);
  }
  panel.textContent = `${message} 実行中のEmbedding Jobがある場合は、完了後に次の処理へ進みます。この画面で進捗表示を更新します。`;
}

function openSelectedAssistLogs() {
  document.querySelectorAll('input[form="epoch-matrix-form"][name="run_ids"]:checked').forEach((checkbox) => {
    const details = document.querySelector(`[data-assist-log-run="${checkbox.value}"]`);
    if (details) {
      details.open = true;
    }
  });
}

async function pollValidationAssistLogs() {
  const panels = [...document.querySelectorAll("[data-assist-log-run]")];
  if (!panels.length) {
    return;
  }
  const payloadsByRunId = new Map();
  await Promise.all(panels.map(async (panel) => {
    const runId = panel.getAttribute("data-assist-log-run");
    if (!runId) {
      return;
    }
    try {
      const response = await fetch(`/validation-runs/${runId}/assist/status`, { headers: { "Accept": "application/json" } });
      if (!response.ok) {
        return;
      }
      const payload = await response.json();
      payloadsByRunId.set(runId, payload);
      const size = panel.querySelector("[data-assist-log-size]");
      const updated = panel.querySelector("[data-assist-log-updated]");
      const preview = panel.querySelector("[data-assist-log-preview]");
      if (size) size.textContent = payload.log_size ?? "0";
      if (updated) updated.textContent = payload.log_updated_at || "-";
      if (preview) {
        preview.classList.toggle("empty", !payload.log_preview);
        preview.textContent = payload.log_preview || "ログはまだありません。一括計算を開始するとここに進行状況が表示されます。";
      }
    } catch (_error) {
      // Polling is informational only; keep the current page stable.
    }
  }));
  finishBulkAssistIfComplete(payloadsByRunId);
}

function validationAssistPayloadIsTerminal(payload) {
  if (!payload) {
    return false;
  }
  const terminalStatuses = ["completed", "failed", "stopped"];
  if (payload.machine_review && terminalStatuses.includes(payload.machine_review.status)) {
    return true;
  }
  if (!payload.machine_review && payload.embedding && ["failed", "stopped"].includes(payload.embedding.status)) {
    return true;
  }
  const logPreview = payload.log_preview || "";
  return /Machine Review Job #\d+: status=(completed|failed|stopped)/.test(logPreview);
}

function finishBulkAssistIfComplete(payloadsByRunId) {
  const form = document.querySelector("[data-bulk-assist-form][data-bulk-assist-running='1']");
  if (!form) {
    return;
  }
  const runIds = (form.dataset.bulkAssistRunIds || "").split(",").filter(Boolean);
  if (!runIds.length) {
    return;
  }
  const isComplete = runIds.every((runId) => validationAssistPayloadIsTerminal(payloadsByRunId.get(runId)));
  if (!isComplete) {
    return;
  }

  delete form.dataset.bulkAssistRunning;
  delete form.dataset.bulkAssistRunIds;
  document.querySelectorAll("#validation-runs button").forEach((candidate) => {
    if (candidate.type === "submit") {
      candidate.disabled = false;
    }
  });
  const button = form.querySelector("[data-bulk-assist-button]");
  if (button) {
    button.textContent = button.dataset.originalText || "画像生成済みRunの不足レビューだけ再計算";
  }
  const panel = document.querySelector("[data-bulk-assist-status]");
  if (panel) {
    panel.textContent = "Embedding / 機械補助レビューが完了しました。必要ならEpoch横断Matrixを開いて確認できます。";
  }
  showPageNotice("Embedding / 機械補助レビューが完了しました。", "success", form);
  if (typeof syncAllGenerationRunButtons === "function") {
    syncAllGenerationRunButtons();
  }
}

function initValidationAssistLogPolling() {
  if (!document.querySelector("[data-assist-log-run]")) {
    return;
  }
  pollValidationAssistLogs();
  window.setInterval(pollValidationAssistLogs, 5000);
}

function appendSelectedValidationRunInputs(form) {
  form.querySelectorAll('input[type="hidden"][name="run_ids"]').forEach((input) => input.remove());
  const checked = [...document.querySelectorAll('input[form="epoch-matrix-form"][name="run_ids"]:checked')];
  checked.forEach((checkbox) => {
    const input = document.createElement("input");
    input.type = "hidden";
    input.name = "run_ids";
    input.value = checkbox.value;
    form.append(input);
  });
  return checked;
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
  const wasRunning = panel.getAttribute("data-generation-status") === "running";
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
  if (wasRunning && ["completed", "failed", "stopped"].includes(payload.status || "")) {
    const message = payload.status === "completed"
      ? "検証画像生成が完了しました。画像とカバレッジを更新します。"
      : "検証画像生成が終了しました。状態を更新します。";
    showPageNotice(message, payload.status === "completed" ? "info" : "warning");
    schedulePageRefresh({ paramsToDelete: ["generation_error"], delayMs: 900 });
  }
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
  const elapsedText = panel.querySelector('[data-operation-field="elapsed"]');
  const stageElapsedText = panel.querySelector('[data-operation-field="stage_elapsed"]');
  const estimatedTotalText = panel.querySelector('[data-operation-field="estimated_total"]');
  const estimatedRemainingText = panel.querySelector('[data-operation-field="estimated_remaining"]');
  const completionEtaText = panel.querySelector('[data-operation-field="completion_eta"]');
  const rateText = panel.querySelector('[data-operation-field="rate"]');
  const logUpdateText = panel.querySelector('[data-operation-field="last_log_update"]');
  const shortLog = panel.querySelector("[data-operation-log-short]");
  const fullLog = panel.querySelector("[data-operation-log-full]");
  const warning = panel.querySelector("[data-operation-log-warning]");
  const followupFields = {
    estimated: panel.querySelector('[data-followup-field="estimated"]'),
    generation: panel.querySelector('[data-followup-field="generation"]'),
    import: panel.querySelector('[data-followup-field="import"]'),
    embedding: panel.querySelector('[data-followup-field="embedding"]'),
    machineReview: panel.querySelector('[data-followup-field="machine_review"]'),
    matrix: panel.querySelector('[data-followup-field="matrix"]'),
    pcTotalRemaining: panel.querySelector('[data-followup-field="pc_total_remaining"]'),
    pcTotalEta: panel.querySelector('[data-followup-field="pc_total_eta"]'),
    basis: panel.querySelector('[data-followup-field="basis"]'),
  };

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
    if (elapsedText && payload.elapsed_label !== undefined) elapsedText.textContent = payload.elapsed_label || "-";
    if (stageElapsedText && payload.stage_elapsed_label !== undefined) stageElapsedText.textContent = payload.stage_elapsed_label || "-";
    if (estimatedTotalText && payload.estimated_total_label !== undefined) estimatedTotalText.textContent = payload.estimated_total_label || "-";
    if (estimatedRemainingText && payload.estimated_remaining_label !== undefined) estimatedRemainingText.textContent = payload.estimated_remaining_label || "-";
    if (completionEtaText && payload.completion_eta_label !== undefined) completionEtaText.textContent = payload.completion_eta_label || "-";
    if (rateText && payload.rate_label !== undefined) rateText.textContent = payload.rate_label || "-";
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
    if (payload.followup_estimate) {
      const followup = payload.followup_estimate;
      if (followupFields.estimated) followupFields.estimated.textContent = followup.estimated_label || "-";
      if (followupFields.generation) followupFields.generation.textContent = followup.generation_seconds_label || "-";
      if (followupFields.import) followupFields.import.textContent = followup.import_seconds_label || "-";
      if (followupFields.embedding) followupFields.embedding.textContent = followup.embedding_seconds_label || "-";
      if (followupFields.machineReview) followupFields.machineReview.textContent = followup.machine_review_seconds_label || "-";
      if (followupFields.matrix) followupFields.matrix.textContent = followup.matrix_seconds_label || "-";
      if (followupFields.pcTotalRemaining) followupFields.pcTotalRemaining.textContent = followup.pc_total_remaining_label || "-";
      if (followupFields.pcTotalEta) followupFields.pcTotalEta.textContent = followup.pc_total_completion_eta_label || "-";
      if (followupFields.basis) followupFields.basis.textContent = followup.basis || "";
    }
    const runningStatuses = new Set(["starting", "running", "generating_images", "importing_images", "embedding_images", "machine_reviewing", "building_matrix"]);
    const isRunning = runningStatuses.has(payload.status);
    const wasRunning = panel.getAttribute("data-operation-running") === "1";
    panel.setAttribute("data-operation-running", isRunning ? "1" : "0");
    if (wasRunning && !isRunning) {
      const terminalMessage = payload.status === "completed"
        ? "処理が完了しました。画面を更新します。"
        : "処理が終了しました。画面を更新します。";
      showPageNotice(terminalMessage);
      schedulePageRefresh({ hash: panel.id ? `#${panel.id}` : "", delayMs: 900 });
    }
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

function initLazySections() {
  document.querySelectorAll("[data-lazy-section]").forEach((section) => {
    const button = section.querySelector("[data-lazy-load-button]");
    const content = section.querySelector("[data-lazy-content]");
    const url = section.getAttribute("data-lazy-url");
    if (!button || !content || !url) {
      return;
    }
    const load = async () => {
      if (section.getAttribute("data-lazy-loaded") === "1") {
        return;
      }
      button.disabled = true;
      const originalText = button.textContent;
      button.textContent = "読み込み中...";
      try {
        const response = await fetch(url, { headers: { "Accept": "text/html" } });
        if (!response.ok) {
          throw new Error(await response.text());
        }
        content.innerHTML = await response.text();
        section.setAttribute("data-lazy-loaded", "1");
        button.remove();
      } catch (error) {
        button.disabled = false;
        button.textContent = originalText || "再読み込み";
        content.innerHTML = `<p class="notice warning">読み込みに失敗しました: ${escapeHtml(error.message)}</p>`;
      }
    };
    button.addEventListener("click", load);
    if (section.hasAttribute("data-lazy-autoload")) {
      load();
    }
  });
}

document.addEventListener("DOMContentLoaded", initValidationGenerationPolling);
document.addEventListener("DOMContentLoaded", initValidationGenerationDetailPolling);
document.addEventListener("DOMContentLoaded", initBulkValidationGenerationSubmit);
document.addEventListener("DOMContentLoaded", initBulkValidationAssistSubmit);
document.addEventListener("DOMContentLoaded", initValidationAssistLogPolling);
document.addEventListener("DOMContentLoaded", clearTransientNoticeParams);
document.addEventListener("DOMContentLoaded", restoreScrollAfterInlineRefresh);
document.addEventListener("DOMContentLoaded", initProjectModeForms);
document.addEventListener("DOMContentLoaded", initDatasetVersionFilters);
document.addEventListener("DOMContentLoaded", initRecipeSelectors);
document.addEventListener("DOMContentLoaded", initStepEstimators);
document.addEventListener("DOMContentLoaded", initReviewAutomationSettings);
document.addEventListener("DOMContentLoaded", initEmbeddingJobStatusPolling);
document.addEventListener("DOMContentLoaded", initMachineReviewJobStatusPolling);
document.addEventListener("DOMContentLoaded", initReviewPreparationPolling);
document.addEventListener("DOMContentLoaded", initTrainLogPolling);
document.addEventListener("DOMContentLoaded", initActiveOperationMonitorPolling);
document.addEventListener("DOMContentLoaded", initLazySections);
