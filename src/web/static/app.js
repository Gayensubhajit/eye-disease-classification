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
