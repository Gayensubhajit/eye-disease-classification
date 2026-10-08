/**
 * Retinal Disease Clinical Screening Studio - Client Application Logic
 * Department of Information Technology, Jadavpur University
 */

let activeBenchmark = "10class";
let modelsRegistry = {};
let allSamples = [];
let currentFile = null;
let currentSampleId = null;
let magnifierActive = false;
let analyticsLoaded = false;

// ================= DOM Elements =================
const gpuNameEl = document.getElementById("gpu-name");
const vramInfoEl = document.getElementById("vram-info");
const headerSotaBadge = document.getElementById("header-sota-badge");

// Navigation Tabs
const navTabs = document.querySelectorAll(".nav-tab-btn");
const tabPanes = document.querySelectorAll(".tab-pane");

// Ribbon Controls
const bench10Btn = document.getElementById("btn-bench-10class");
const bench4Btn = document.getElementById("btn-bench-4class");
const modelSelect = document.getElementById("model-select");
const chkGradcam = document.getElementById("chk-gradcam");

// Sample Picker
const sampleGallery = document.getElementById("sample-gallery-container");
const sampleFilterChips = document.querySelectorAll(".sample-filter-chips .filter-chip");

// Viewport Elements
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const previewContainer = document.getElementById("preview-container");
const fundusStage = document.getElementById("fundus-stage");
const fundusImgBase = document.getElementById("fundus-img-base");
const fundusImgOverlay = document.getElementById("fundus-img-overlay");
const splitLayerClip = document.getElementById("split-layer-clip");
const splitDivider = document.getElementById("split-divider");
const magnifierLoupe = document.getElementById("magnifier-loupe");
const scanningLaser = document.getElementById("scanning-laser");
const previewBadge = document.getElementById("preview-badge");
const viewportStatus = document.getElementById("viewport-status");
const btnClearImg = document.getElementById("btn-clear-img");
const btnAnalyze = document.getElementById("btn-analyze");
const btnAnalyzeText = document.getElementById("btn-analyze-text");
const analyzeSpinner = document.getElementById("analyze-spinner");
const analyzeIcon = document.getElementById("analyze-icon");
const btnToggleMagnifier = document.getElementById("btn-toggle-magnifier");
const optFilterBtns = document.querySelectorAll(".opt-filter-btn");

// Results HUD Elements
const resultsPlaceholder = document.getElementById("results-placeholder");
const resultsContainer = document.getElementById("results-container");
const resModelBadge = document.getElementById("res-model-badge");
const resLatency = document.getElementById("res-latency");
const resTopDisease = document.getElementById("res-top-disease");
const resUrgencyPill = document.getElementById("res-urgency-pill");
const resUrgencyBeacon = document.getElementById("res-urgency-beacon");
const resUrgencyText = document.getElementById("res-urgency-text");
const resCertaintyPill = document.getElementById("res-certainty-pill");
const gaugeFillCircle = document.getElementById("gauge-fill-circle");
const resConfidenceNum = document.getElementById("res-confidence-num");
const entropyBarFill = document.getElementById("entropy-bar-fill");
const resEntropyVal = document.getElementById("res-entropy-val");
const safetyStatusBadge = document.getElementById("safety-status-badge");
const resNumClasses = document.getElementById("res-num-classes");
const probBarsContainer = document.getElementById("prob-bars-container");
const resDiseaseDesc = document.getElementById("res-disease-desc");
const resRecommendationsList = document.getElementById("res-recommendations-list");
const btnExportReport = document.getElementById("btn-export-report");

// Mode Toggle & Workspace Sections
const modeMonocularBtn = document.getElementById("btn-mode-monocular");
const modeBilateralBtn = document.getElementById("btn-mode-bilateral");
const monocularWorkspace = document.getElementById("monocular-workspace");
const bilateralWorkspace = document.getElementById("bilateral-workspace");
const samplePickerBar = document.querySelector(".sample-picker-bar");

// Conformal Set DOM
const confCutoff = document.getElementById("conf-cutoff");
const confStatusTag = document.getElementById("conf-status-tag");
const confBadgeText = document.getElementById("conf-badge-text");
const confSetSize = document.getElementById("conf-set-size");
const confCoverage = document.getElementById("conf-coverage");
const confClinicalStatus = document.getElementById("conf-clinical-status");
const confChipsContainer = document.getElementById("conf-chips-container");
const confActionText = document.getElementById("conf-action-text");

// Bilateral DOM
const odDropZone = document.getElementById("od-drop-zone");
const odFileInput = document.getElementById("od-file-input");
const odPreviewStage = document.getElementById("od-preview-stage");
const odPreviewImg = document.getElementById("od-preview-img");
const odLabel = document.getElementById("od-label");
const odStatus = document.getElementById("od-status");
const btnChangeOd = document.getElementById("btn-change-od");

const osDropZone = document.getElementById("os-drop-zone");
const osFileInput = document.getElementById("os-file-input");
const osPreviewStage = document.getElementById("os-preview-stage");
const osPreviewImg = document.getElementById("os-preview-img");
const osLabel = document.getElementById("os-label");
const osStatus = document.getElementById("os-status");
const btnChangeOs = document.getElementById("btn-change-os");

const btnAnalyzeBilateral = document.getElementById("btn-analyze-bilateral");
const bilateralSpinner = document.getElementById("bilateral-spinner");
const bilateralIcon = document.getElementById("bilateral-icon");
const btnBilateralText = document.getElementById("btn-bilateral-text");
const bilateralResultsContainer = document.getElementById("bilateral-results-container");

const baiStatusBadge = document.getElementById("bai-status-badge");
const baiAsymmetryVal = document.getElementById("bai-asymmetry-val");
const baiMeterBar = document.getElementById("bai-meter-bar");
const baiCategoryTitle = document.getElementById("bai-category-title");
const baiSummaryText = document.getElementById("bai-summary-text");
const baiActionText = document.getElementById("bai-action-text");

const odResLatency = document.getElementById("od-res-latency");
const odResDisease = document.getElementById("od-res-disease");
const odResConf = document.getElementById("od-res-conf");
const odResConformalChips = document.getElementById("od-res-conformal-chips");
const odResCamImg = document.getElementById("od-res-cam-img");

const osResLatency = document.getElementById("os-res-latency");
const osResDisease = document.getElementById("os-res-disease");
const osResConf = document.getElementById("os-res-conf");
const osResConformalChips = document.getElementById("os-res-conformal-chips");
const osResCamImg = document.getElementById("os-res-cam-img");

const presetBtns = document.querySelectorAll(".btn-preset-case");

let activeMode = "monocular";
let currentOdFile = null;
let currentOdSampleId = "dr_proliferative";
let currentOsFile = null;
let currentOsSampleId = "dr_proliferative";

// ================= Initialization =================
document.addEventListener("DOMContentLoaded", async () => {
  await fetchTelemetry();
  await fetchModels();
  await fetchSamples();
  setupNavigationTabs();
  setupControlRibbon();
  setupDropZone();
  setupSplitSlider();
  setupMagnifierLoupe();
  setupOpticalFilters();
  setupReportExport();
  setupModeToggle();
  setupBilateralMode();
  setInterval(fetchTelemetry, 12000);
});

// ================= Navigation Tabs =================
function setupNavigationTabs() {
  navTabs.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetTabId = btn.dataset.tab;
      navTabs.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(p => p.style.display = "none");

      btn.classList.add("active");
      const targetPane = document.getElementById(targetTabId);
      if (targetPane) {
        targetPane.style.display = "block";
      }

      if (targetTabId === "tab-benchmarks" && !analyticsLoaded) {
        loadAnalyticsTable();
      }
    });
  });
}

// ================= Telemetry =================
async function fetchTelemetry() {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) return;
    const data = await res.json();
    gpuNameEl.textContent = data.gpu_name || "CPU (Host)";
    if (data.cuda_available) {
      vramInfoEl.textContent = `| VRAM: ${data.vram_allocated_mb} MB / ${data.vram_reserved_mb} MB`;
    } else {
      vramInfoEl.textContent = "";
    }
  } catch (err) {
    console.error("Telemetry fetch error:", err);
  }
}

// ================= Models =================
async function fetchModels() {
  try {
    const res = await fetch("/api/models");
    if (!res.ok) return;
    modelsRegistry = await res.json();
    populateModelOptions();
  } catch (err) {
    console.error("Models fetch error:", err);
  }
}

function populateModelOptions() {
  modelSelect.innerHTML = "";
  const benchData = modelsRegistry[activeBenchmark];
  if (!benchData || !benchData.models) return;

  const models = benchData.models;
  for (const [key, item] of Object.entries(models)) {
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = `${item.name} (${item.accuracy})`;
    modelSelect.appendChild(opt);
  }
}

// ================= Curated Samples =================
async function fetchSamples() {
  try {
    const res = await fetch("/api/samples");
    if (!res.ok) return;
    allSamples = await res.json();
    renderSampleCards(allSamples);
    setupSampleFilters();
  } catch (err) {
    console.error("Samples fetch error:", err);
  }
}

function renderSampleCards(samplesToRender) {
  sampleGallery.innerHTML = "";
  samplesToRender.forEach(sample => {
    const card = document.createElement("div");
    card.className = "sample-card";
    card.dataset.sampleId = sample.id;
    card.dataset.benchmark = sample.benchmark || "10class";

    const tagColor = sample.benchmark === "4class" ? "rgba(168, 85, 247, 0.25)" : "rgba(0, 242, 254, 0.2)";
    const tagBorder = sample.benchmark === "4class" ? "#c084fc" : "#00f2fe";

    card.innerHTML = `
      <div class="sample-card-img-wrap">
        <img src="${sample.image_url}" alt="${sample.title}" loading="lazy">
        <span class="sample-tag" style="background: ${tagColor}; border: 1px solid ${tagBorder}; color: #fff;">
          ${sample.benchmark === "4class" ? "4-Class" : "10-Class"}
        </span>
      </div>
      <div class="sample-card-title">${sample.title}</div>
      <div class="sample-card-sub">${sample.subtitle}</div>
    `;

    card.addEventListener("click", () => selectSample(sample, card));
    sampleGallery.appendChild(card);
  });
}

function setupSampleFilters() {
  sampleFilterChips.forEach(chip => {
    chip.addEventListener("click", () => {
      sampleFilterChips.forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      const filter = chip.dataset.filter;

      if (filter === "all") {
        renderSampleCards(allSamples);
      } else if (filter === "sight_threatening") {
        const filtered = allSamples.filter(s => 
          s.id.includes("dr") || s.id.includes("glaucoma") || s.id.includes("detachment") || s.id.includes("edema")
        );
        renderSampleCards(filtered);
      } else if (filter === "macular") {
        const filtered = allSamples.filter(s => s.id.includes("cscr") || s.id.includes("dr"));
        renderSampleCards(filtered);
      } else if (filter === "normal") {
        const filtered = allSamples.filter(s => s.id.includes("healthy") || s.id.includes("normal") || s.id.includes("cataract"));
        renderSampleCards(filtered);
      }
    });
  });
}

async function selectSample(sample, cardEl) {
  document.querySelectorAll(".sample-card").forEach(c => c.classList.remove("active"));
  cardEl.classList.add("active");

  currentSampleId = sample.id;
  currentFile = null;

  // Auto-align benchmark if needed
  if (sample.benchmark && sample.benchmark !== activeBenchmark) {
    setBenchmark(sample.benchmark);
  }

  // Load image into Fundus Viewport
  fundusImgBase.src = sample.image_url;
  fundusImgOverlay.src = sample.image_url;
  magnifierLoupe.style.backgroundImage = `url('${sample.image_url}')`;

  previewBadge.textContent = `${sample.title} (Cohort)`;
  dropZone.style.display = "none";
  previewContainer.style.display = "block";
  viewportStatus.textContent = "Cohort Scan Loaded";
  viewportStatus.className = "status-chip ready-chip";

  resetSplitSlider();

  // Trigger analysis immediately
  executeInference();
}

// ================= Benchmark Switcher =================
function setupControlRibbon() {
  bench10Btn.addEventListener("click", () => setBenchmark("10class"));
  bench4Btn.addEventListener("click", () => setBenchmark("4class"));
}

function setBenchmark(bench) {
  activeBenchmark = bench;
  if (bench === "10class") {
    bench10Btn.classList.add("active");
    bench4Btn.classList.remove("active");
    headerSotaBadge.textContent = "10-Class (91.83%)";
  } else {
    bench4Btn.classList.add("active");
    bench10Btn.classList.remove("active");
    headerSotaBadge.textContent = "4-Class (95.74%)";
  }
  populateModelOptions();
}

// ================= Drop Zone & File Input =================
function setupDropZone() {
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleCustomFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleCustomFile(e.target.files[0]);
    }
  });

  btnClearImg.addEventListener("click", clearWorkspace);
  btnAnalyze.addEventListener("click", executeInference);
}

function handleCustomFile(file) {
  if (!file.type.startsWith("image/")) {
    alert("Please upload a valid fundus image file (JPEG, PNG).");
    return;
  }

  currentFile = file;
  currentSampleId = null;
  document.querySelectorAll(".sample-card").forEach(c => c.classList.remove("active"));

  const reader = new FileReader();
  reader.onload = (e) => {
    const dataUrl = e.target.result;
    fundusImgBase.src = dataUrl;
    fundusImgOverlay.src = dataUrl;
    magnifierLoupe.style.backgroundImage = `url('${dataUrl}')`;

    previewBadge.textContent = `Patient File: ${file.name.slice(0, 18)}...`;
    dropZone.style.display = "none";
    previewContainer.style.display = "block";
    viewportStatus.textContent = "Patient Image Staged";
    viewportStatus.className = "status-chip ready-chip";

    resetSplitSlider();
    executeInference();
  };
  reader.readAsDataURL(file);
}

function clearWorkspace() {
  currentFile = null;
  currentSampleId = null;
  fileInput.value = "";
  fundusImgBase.src = "";
  fundusImgOverlay.src = "";
  previewContainer.style.display = "none";
  dropZone.style.display = "block";
  resultsContainer.style.display = "none";
  resultsPlaceholder.style.display = "block";
  viewportStatus.textContent = "Ready for Input";
  document.querySelectorAll(".sample-card").forEach(c => c.classList.remove("active"));
}

// ================= Interactive Before / After Split Slider =================
let isDraggingSplit = false;

function setupSplitSlider() {
  splitDivider.addEventListener("mousedown", () => isDraggingSplit = true);
  window.addEventListener("mouseup", () => isDraggingSplit = false);
  window.addEventListener("mousemove", handleSplitMove);

  splitDivider.addEventListener("touchstart", () => isDraggingSplit = true);
  window.addEventListener("touchend", () => isDraggingSplit = false);
  window.addEventListener("touchmove", (e) => {
    if (isDraggingSplit && e.touches.length > 0) {
      handleSplitMove(e.touches[0]);
    }
  });
}

function handleSplitMove(e) {
  if (!isDraggingSplit) return;
  const rect = fundusStage.getBoundingClientRect();
  let clientX = e.clientX;
  let offsetX = clientX - rect.left;
  let percent = (offsetX / rect.width) * 100;

  if (percent < 5) percent = 5;
  if (percent > 95) percent = 95;

  splitDivider.style.left = `${percent}%`;
  splitLayerClip.style.clipPath = `inset(0 0 0 ${percent}%)`;
}

function resetSplitSlider() {
  splitDivider.style.left = "50%";
  splitLayerClip.style.clipPath = "inset(0 0 0 50%)";
}

// ================= 2.5x Loupe Magnifier =================
function setupMagnifierLoupe() {
  btnToggleMagnifier.addEventListener("click", () => {
    magnifierActive = !magnifierActive;
    btnToggleMagnifier.classList.toggle("active", magnifierActive);
    if (!magnifierActive) {
      magnifierLoupe.style.display = "none";
    }
  });

  fundusStage.addEventListener("mousemove", (e) => {
    if (!magnifierActive) return;
    magnifierLoupe.style.display = "block";

    const rect = fundusStage.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const loupeSize = 140;
    magnifierLoupe.style.left = `${x - loupeSize / 2}px`;
    magnifierLoupe.style.top = `${y - loupeSize / 2}px`;

    const zoom = 2.5;
    magnifierLoupe.style.backgroundSize = `${rect.width * zoom}px ${rect.height * zoom}px`;
    magnifierLoupe.style.backgroundPosition = `-${x * zoom - loupeSize / 2}px -${y * zoom - loupeSize / 2}px`;
  });

  fundusStage.addEventListener("mouseleave", () => {
    magnifierLoupe.style.display = "none";
  });
}

// ================= Optical Ophthalmic Filters =================
function setupOpticalFilters() {
  optFilterBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      optFilterBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const filter = btn.dataset.filter;

      fundusImgBase.className = "fundus-layer base-layer";
      fundusImgOverlay.className = "fundus-layer overlay-layer";

      if (filter === "redfree") {
        fundusImgBase.classList.add("filter-redfree");
        fundusImgOverlay.classList.add("filter-redfree");
      } else if (filter === "clahe") {
        fundusImgBase.classList.add("filter-clahe");
        fundusImgOverlay.classList.add("filter-clahe");
      } else if (filter === "invert") {
        fundusImgBase.classList.add("filter-invert");
        fundusImgOverlay.classList.add("filter-invert");
      }
    });
  });
}

// ================= Inference Execution & Animated Findings HUD =================
async function executeInference() {
  if (!currentFile && !currentSampleId) return;

  setAnalyzingState(true);

  try {
    const formData = new FormData();
    formData.append("benchmark", activeBenchmark);
    formData.append("model_id", modelSelect.value);
    formData.append("generate_cam", chkGradcam.checked ? "true" : "false");

    if (currentFile) {
      formData.append("file", currentFile);
    } else {
      formData.append("sample_id", currentSampleId);
    }

    const res = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Inference failed");
    }

    const data = await res.json();
    renderDiagnosticFindings(data);
  } catch (err) {
    alert(`Diagnosis Error: ${err.message}`);
  } finally {
    setAnalyzingState(false);
  }
}

function setAnalyzingState(isAnalyzing) {
  if (isAnalyzing) {
    btnAnalyze.disabled = true;
    analyzeSpinner.style.display = "block";
    analyzeIcon.style.display = "none";
    btnAnalyzeText.textContent = "Analyzing Retinal Micro-structures...";
    scanningLaser.classList.add("active");
    viewportStatus.textContent = "AI Inference Active...";
  } else {
    btnAnalyze.disabled = false;
    analyzeSpinner.style.display = "none";
    analyzeIcon.style.display = "block";
    btnAnalyzeText.textContent = "Re-Evaluate Screening";
    scanningLaser.classList.remove("active");
    viewportStatus.textContent = "Diagnostic Assessment Complete";
  }
}

function renderDiagnosticFindings(data) {
  resultsPlaceholder.style.display = "none";
  resultsContainer.style.display = "flex";

  const top = data.top_prediction;
  const unc = data.uncertainty;

  // Header badges
  resModelBadge.textContent = `${data.model_name}`;
  resLatency.textContent = `⏱️ ${data.latency_ms} ms`;
  resTopDisease.textContent = top.short_name || top.class_name;

  // Urgency Beacon
  resUrgencyText.textContent = `${top.urgency} Urgency`;
  if (top.urgency_level === "red") {
    resUrgencyPill.style.background = "rgba(244, 63, 94, 0.18)";
    resUrgencyPill.style.color = "#fb7185";
    resUrgencyPill.style.borderColor = "rgba(244, 63, 94, 0.4)";
    resUrgencyBeacon.style.backgroundColor = "#fb7185";
  } else if (top.urgency_level === "amber") {
    resUrgencyPill.style.background = "rgba(245, 158, 11, 0.18)";
    resUrgencyPill.style.color = "#fbbf24";
    resUrgencyPill.style.borderColor = "rgba(245, 158, 11, 0.4)";
    resUrgencyBeacon.style.backgroundColor = "#fbbf24";
  } else {
    resUrgencyPill.style.background = "rgba(16, 185, 129, 0.15)";
    resUrgencyPill.style.color = "#34d399";
    resUrgencyPill.style.borderColor = "rgba(16, 185, 129, 0.35)";
    resUrgencyBeacon.style.backgroundColor = "#34d399";
  }

  // Certainty Pill
  resCertaintyPill.textContent = `${unc.certainty_badge} Consensus`;

  // Animated Circular SVG Radial Gauge
  const targetPct = top.percentage;
  animateConfidenceCounter(targetPct);
  const circumference = 314.16;
  const strokeOffset = circumference - (circumference * (targetPct / 100));
  gaugeFillCircle.style.strokeDashoffset = strokeOffset;

  // Shannon Entropy Uncertainty Strip
  const normEntropy = unc.normalized_entropy !== undefined ? unc.normalized_entropy : unc.entropy;
  resEntropyVal.textContent = `${normEntropy.toFixed(3)} / 1.000`;
  const entropyPct = Math.min(Math.max(normEntropy * 100, 5), 100);
  entropyBarFill.style.width = `${entropyPct}%`;

  if (normEntropy < 0.25) {
    safetyStatusBadge.className = "safety-status-badge cleared";
    safetyStatusBadge.textContent = "✅ Cleared (Low Entropy)";
  } else if (normEntropy < 0.40) {
    safetyStatusBadge.className = "safety-status-badge flagged";
    safetyStatusBadge.textContent = "⚠️ Borderline (Resident Review)";
  } else {
    safetyStatusBadge.className = "safety-status-badge flagged";
    safetyStatusBadge.style.color = "#f87171";
    safetyStatusBadge.textContent = "🚨 High Uncertainty (OCT Escalation)";
  }

  // Conformal Prediction Risk Set Rendering
  if (data.conformal && confChipsContainer) {
    const conf = data.conformal;
    if (confCutoff) confCutoff.textContent = `p ≥ ${conf.probability_cutoff}`;
    if (confSetSize) confSetSize.textContent = `${conf.set_size} ${conf.set_size === 1 ? 'Pathology' : 'Pathologies'}`;
    if (confCoverage) confCoverage.textContent = conf.empirical_test_coverage;
    if (confClinicalStatus) confClinicalStatus.textContent = conf.clinical_status;
    if (confBadgeText) confBadgeText.textContent = conf.clinical_badge;
    if (confStatusTag) confStatusTag.className = `conformal-status-tag ${conf.badge_color}`;
    if (confActionText) confActionText.textContent = conf.clinical_action;

    confChipsContainer.innerHTML = "";
    conf.prediction_set.forEach((clsName, idx) => {
      const p = conf.prediction_set_probabilities[idx];
      const chip = document.createElement("div");
      chip.className = "conformal-chip";
      chip.innerHTML = `<span>${clsName}</span><span class="conformal-chip-prob">${p}%</span>`;
      confChipsContainer.appendChild(chip);
    });
  }

  // Grad-CAM Layer Update
  if (data.gradcam && data.gradcam.overlay_b64) {
    fundusImgOverlay.src = data.gradcam.overlay_b64;
    if (data.gradcam.preprocessed_b64) {
      fundusImgBase.src = data.gradcam.preprocessed_b64;
    }
  }

  // Differential Diagnostic Spectrum Bars
  resNumClasses.textContent = `${data.distribution.length} Pathologies Evaluated`;
  probBarsContainer.innerHTML = "";
  data.distribution.forEach(item => {
    const isTop = item.rank === 1;
    const row = document.createElement("div");
    row.className = `prob-item ${isTop ? "is-top" : ""}`;
    row.innerHTML = `
      <div class="prob-meta">
        <span class="prob-disease-name">${item.class_name}</span>
        <span class="prob-percentage">${item.percentage.toFixed(2)}%</span>
      </div>
      <div class="prob-track">
        <div class="prob-bar" style="width: 0%;"></div>
      </div>
    `;
    probBarsContainer.appendChild(row);

    setTimeout(() => {
      const bar = row.querySelector(".prob-bar");
      if (bar) bar.style.width = `${Math.max(item.percentage, 1)}%`;
    }, 50);
  });

  // Clinical Guidance & Recommended Follow-Up
  resDiseaseDesc.textContent = top.description;
  resRecommendationsList.innerHTML = "";
  if (top.recommendations && top.recommendations.length > 0) {
    top.recommendations.forEach(rec => {
      const li = document.createElement("li");
      li.textContent = rec;
      resRecommendationsList.appendChild(li);
    });
  }
}

function animateConfidenceCounter(target) {
  let current = 0;
  const duration = 650;
  const stepTime = 20;
  const steps = duration / stepTime;
  const increment = target / steps;

  const timer = setInterval(() => {
    current += increment;
    if (current >= target) {
      current = target;
      clearInterval(timer);
    }
    resConfidenceNum.textContent = `${current.toFixed(1)}%`;
  }, stepTime);
}

// ================= Tab 2: Dynamic Analytics Table =================
async function loadAnalyticsTable() {
  try {
    const res = await fetch("/api/analytics");
    if (!res.ok) return;
    const data = await res.json();
    const metrics10 = data.benchmark_10class.per_class_metrics;
    const tbody = document.getElementById("metrics-table-body");
    if (!tbody) return;

    tbody.innerHTML = "";
    metrics10.forEach(m => {
      const isFlawless = m.false_negatives === 0;
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${m.class_name.replace(" [Color Fundus]", "")}</strong></td>
        <td>${m.total_samples}</td>
        <td class="${isFlawless ? 'highlight-flawless' : ''}">${m.true_positives}</td>
        <td class="${isFlawless ? 'highlight-flawless' : ''}">${m.false_negatives}</td>
        <td>${m.false_positives}</td>
        <td class="${isFlawless ? 'highlight-flawless' : ''}"><strong>${(m.sensitivity * 100).toFixed(1)}%</strong></td>
        <td>${(m.specificity * 100).toFixed(1)}%</td>
        <td>${(m.precision * 100).toFixed(1)}%</td>
        <td><strong>${(m.f1_score * 100).toFixed(1)}%</strong></td>
      `;
      tbody.appendChild(tr);
    });
    analyticsLoaded = true;
  } catch (err) {
    console.error("Analytics table error:", err);
  }
}

// ================= Printable Report =================
function setupReportExport() {
  btnExportReport.addEventListener("click", () => {
    window.print();
  });
}


// ================= Inspection Mode (Monocular vs Bilateral) =================
function setupModeToggle() {
  if (!modeMonocularBtn || !modeBilateralBtn) return;
  modeMonocularBtn.addEventListener("click", () => setMode("monocular"));
  modeBilateralBtn.addEventListener("click", () => setMode("bilateral"));
}

function setMode(mode) {
  activeMode = mode;
  if (mode === "monocular") {
    modeMonocularBtn.classList.add("active");
    modeBilateralBtn.classList.remove("active");
    monocularWorkspace.style.display = "grid";
    bilateralWorkspace.style.display = "none";
    if (samplePickerBar) samplePickerBar.style.display = "block";
  } else {
    modeBilateralBtn.classList.add("active");
    modeMonocularBtn.classList.remove("active");
    monocularWorkspace.style.display = "none";
    bilateralWorkspace.style.display = "flex";
    if (samplePickerBar) samplePickerBar.style.display = "none";
    initBilateralPreset("dr_proliferative", "dr_proliferative");
  }
}

// ================= Bilateral Dual-Eye Screening Logic =================
function setupBilateralMode() {
  if (!btnAnalyzeBilateral) return;

  presetBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      presetBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const odId = btn.dataset.od;
      const osId = btn.dataset.os;
      initBilateralPreset(odId, osId);
    });
  });

  if (btnChangeOd) {
    btnChangeOd.addEventListener("click", () => odFileInput.click());
  }
  if (odFileInput) {
    odFileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleBilateralFile("OD", e.target.files[0]);
      }
    });
  }
  if (odDropZone) {
    odDropZone.addEventListener("click", () => odFileInput.click());
    odDropZone.addEventListener("dragover", (e) => { e.preventDefault(); odDropZone.classList.add("dragover"); });
    odDropZone.addEventListener("dragleave", () => odDropZone.classList.remove("dragover"));
    odDropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      odDropZone.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleBilateralFile("OD", e.dataTransfer.files[0]);
      }
    });
  }

  if (btnChangeOs) {
    btnChangeOs.addEventListener("click", () => osFileInput.click());
  }
  if (osFileInput) {
    osFileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleBilateralFile("OS", e.target.files[0]);
      }
    });
  }
  if (osDropZone) {
    osDropZone.addEventListener("click", () => osFileInput.click());
    osDropZone.addEventListener("dragover", (e) => { e.preventDefault(); osDropZone.classList.add("dragover"); });
    osDropZone.addEventListener("dragleave", () => osDropZone.classList.remove("dragover"));
    osDropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      osDropZone.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleBilateralFile("OS", e.dataTransfer.files[0]);
      }
    });
  }

  btnAnalyzeBilateral.addEventListener("click", executeBilateralInference);
}

function initBilateralPreset(odId, osId) {
  currentOdSampleId = odId;
  currentOdFile = null;
  currentOsSampleId = osId;
  currentOsFile = null;

  const odSample = allSamples.find(s => s.id === odId);
  if (odSample && odPreviewImg) {
    odPreviewImg.src = odSample.image_url;
    odPreviewStage.style.display = "block";
    odDropZone.style.display = "none";
    odLabel.textContent = `OD: ${odSample.title}`;
    odStatus.textContent = "Ready";
  }

  const osSample = allSamples.find(s => s.id === osId);
  if (osSample && osPreviewImg) {
    osPreviewImg.src = osSample.image_url;
    osPreviewStage.style.display = "block";
    osDropZone.style.display = "none";
    osLabel.textContent = `OS: ${osSample.title}`;
    osStatus.textContent = "Ready";
  }
}

function handleBilateralFile(eye, file) {
  if (!file.type.startsWith("image/")) {
    alert("Please upload a valid fundus image file.");
    return;
  }
  const reader = new FileReader();
  reader.onload = (e) => {
    if (eye === "OD") {
      currentOdFile = file;
      currentOdSampleId = null;
      odPreviewImg.src = e.target.result;
      odPreviewStage.style.display = "block";
      odDropZone.style.display = "none";
      odLabel.textContent = `OD: ${file.name}`;
      odStatus.textContent = "Custom Image";
    } else {
      currentOsFile = file;
      currentOsSampleId = null;
      osPreviewImg.src = e.target.result;
      osPreviewStage.style.display = "block";
      osDropZone.style.display = "none";
      osLabel.textContent = `OS: ${file.name}`;
      osStatus.textContent = "Custom Image";
    }
  };
  reader.readAsDataURL(file);
}

async function executeBilateralInference() {
  if ((!currentOdFile && !currentOdSampleId) || (!currentOsFile && !currentOsSampleId)) {
    alert("Please provide both OD (Right Eye) and OS (Left Eye) images.");
    return;
  }

  btnAnalyzeBilateral.disabled = true;
  bilateralSpinner.style.display = "block";
  bilateralIcon.style.display = "none";
  btnBilateralText.textContent = "Comparing Inter-Ocular Morphologies...";

  try {
    const formData = new FormData();
    formData.append("benchmark", activeBenchmark);
    formData.append("model_id", modelSelect.value);
    formData.append("generate_cam", chkGradcam.checked ? "true" : "false");

    if (currentOdFile) {
      formData.append("od_file", currentOdFile);
    } else {
      formData.append("od_sample_id", currentOdSampleId);
    }

    if (currentOsFile) {
      formData.append("os_file", currentOsFile);
    } else {
      formData.append("os_sample_id", currentOsSampleId);
    }

    const res = await fetch("/api/predict-bilateral", {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Bilateral analysis failed");
    }

    const data = await res.json();
    renderBilateralFindings(data);
  } catch (err) {
    alert(`Bilateral Screening Error: ${err.message}`);
  } finally {
    btnAnalyzeBilateral.disabled = false;
    bilateralSpinner.style.display = "none";
    bilateralIcon.style.display = "block";
    btnBilateralText.textContent = "Re-Evaluate Bilateral Comparative Screening";
  }
}

function renderBilateralFindings(data) {
  bilateralResultsContainer.style.display = "flex";

  const analysis = data.bilateral_analysis;
  const od = data.od;
  const os = data.os;

  // Synthesis Hero
  baiStatusBadge.className = `bai-status-badge ${analysis.badge_color || 'emerald'}`;
  baiStatusBadge.textContent = analysis.concordant ? "Concordant Systemic Pathology" : "Discordant Unilateral Asymmetry";

  baiAsymmetryVal.textContent = `${analysis.asymmetry_percentage.toFixed(1)}%`;
  baiMeterBar.style.width = `${Math.min(analysis.asymmetry_percentage, 100)}%`;

  baiCategoryTitle.textContent = analysis.clinical_category;
  baiSummaryText.textContent = analysis.clinical_summary;
  baiActionText.textContent = `Recommended Clinical Protocol: ${analysis.recommended_action}`;

  // OD Column
  odResLatency.textContent = `⏱️ ${od.latency_ms} ms`;
  odResDisease.textContent = od.top_prediction.short_name || od.top_prediction.class_name;
  odResConf.textContent = `${od.top_prediction.percentage.toFixed(1)}% Confidence`;
  if (od.gradcam && od.gradcam.overlay_b64) {
    odResCamImg.src = od.gradcam.overlay_b64;
  }
  odResConformalChips.innerHTML = "";
  if (od.conformal && od.conformal.prediction_set) {
    od.conformal.prediction_set.forEach((cls, i) => {
      const p = od.conformal.prediction_set_probabilities[i];
      const chip = document.createElement("span");
      chip.className = "conformal-chip";
      chip.innerHTML = `${cls} <span class="conformal-chip-prob">${p}%</span>`;
      odResConformalChips.appendChild(chip);
    });
  }

  // OS Column
  osResLatency.textContent = `⏱️ ${os.latency_ms} ms`;
  osResDisease.textContent = os.top_prediction.short_name || os.top_prediction.class_name;
  osResConf.textContent = `${os.top_prediction.percentage.toFixed(1)}% Confidence`;
  if (os.gradcam && os.gradcam.overlay_b64) {
    osResCamImg.src = os.gradcam.overlay_b64;
  }
  osResConformalChips.innerHTML = "";
  if (os.conformal && os.conformal.prediction_set) {
    os.conformal.prediction_set.forEach((cls, i) => {
      const p = os.conformal.prediction_set_probabilities[i];
      const chip = document.createElement("span");
      chip.className = "conformal-chip";
      chip.innerHTML = `${cls} <span class="conformal-chip-prob">${p}%</span>`;
      osResConformalChips.appendChild(chip);
    });
  }

  bilateralResultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
