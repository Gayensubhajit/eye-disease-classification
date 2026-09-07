/**
 * Retinal Disease Clinical Screening Studio - Client Application Logic
 */

let activeBenchmark = "10class";
let modelsRegistry = {};
let currentFile = null;
let currentSampleId = null;

// DOM Elements
const gpuNameEl = document.getElementById("gpu-name");
const vramInfoEl = document.getElementById("vram-info");
const bench10Btn = document.getElementById("btn-bench-10class");
const bench4Btn = document.getElementById("btn-bench-4class");
const modelSelect = document.getElementById("model-select");
const chkGradcam = document.getElementById("chk-gradcam");
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const previewContainer = document.getElementById("preview-container");
const previewImg = document.getElementById("preview-img");
const previewBadge = document.getElementById("preview-badge");
const btnClearImg = document.getElementById("btn-clear-img");
const btnAnalyze = document.getElementById("btn-analyze");
const btnAnalyzeText = document.getElementById("btn-analyze-text");
const sampleGallery = document.getElementById("sample-gallery-container");
const resultsPlaceholder = document.getElementById("results-placeholder");
const resultsContainer = document.getElementById("results-container");

// Results Elements
const resModelBadge = document.getElementById("res-model-badge");
const resTopDisease = document.getElementById("res-top-disease");
const resUrgencyPill = document.getElementById("res-urgency-pill");
const resUrgencyIcon = document.getElementById("res-urgency-icon");
const resUrgencyText = document.getElementById("res-urgency-text");
const resCertaintyPill = document.getElementById("res-certainty-pill");
const resConfidenceNum = document.getElementById("res-confidence-num");
const resEntropyVal = document.getElementById("res-entropy-val");
const resConsensusVal = document.getElementById("res-consensus-val");
const resNumClasses = document.getElementById("res-num-classes");
const probBarsContainer = document.getElementById("prob-bars-container");
const resDiseaseDesc = document.getElementById("res-disease-desc");
const resRecommendationsList = document.getElementById("res-recommendations-list");
const btnExportReport = document.getElementById("btn-export-report");

// Grad-CAM Elements
const cardGradcam = document.getElementById("card-gradcam");
const camImgPreprocessed = document.getElementById("cam-img-preprocessed");
const camImgOverlay = document.getElementById("cam-img-overlay");
const camOpacitySlider = document.getElementById("cam-opacity-slider");

// ================= Initialization =================
document.addEventListener("DOMContentLoaded", async () => {
  await fetchTelemetry();
  await fetchModels();
  await fetchSamples();
  setupEventListeners();
  setInterval(fetchTelemetry, 15000);
});

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
    opt.textContent = `${item.name} (Acc: ${item.accuracy})`;
    modelSelect.appendChild(opt);
  }
}

// ================= Curated Samples =================
async function fetchSamples() {
  try {
    const res = await fetch("/api/samples");
    if (!res.ok) return;
    const samples = await res.json();
    renderSampleCards(samples);
  } catch (err) {
    console.error("Samples fetch error:", err);
  }
}

function renderSampleCards(samples) {
  sampleGallery.innerHTML = "";
  samples.forEach(sample => {
    const card = document.createElement("div");
    card.className = "sample-card";
    card.dataset.sampleId = sample.id;
    card.dataset.benchmark = sample.benchmark;

    card.innerHTML = `
      <img src="${sample.image_url}" alt="${sample.title}" class="sample-thumb" loading="lazy">
      <div class="sample-meta">
        <span class="sample-title">${sample.title}</span>
        <span class="sample-sub">${sample.subtitle}</span>
      </div>
    `;

    card.addEventListener("click", () => {
      selectSample(sample);
    });

    sampleGallery.appendChild(card);
  });
}

function selectSample(sample) {
  currentFile = null;
  currentSampleId = sample.id;

  // Auto switch benchmark if sample belongs to different benchmark
  if (sample.benchmark !== activeBenchmark) {
    switchBenchmark(sample.benchmark);
  }

  // Update preview
  previewImg.src = sample.image_url;
  previewBadge.textContent = `${sample.title} (Test Case)`;
  previewContainer.style.display = "flex";
  dropZone.style.display = "none";
  btnClearImg.style.display = "block";

  // Highlight active sample card
  document.querySelectorAll(".sample-card").forEach(c => {
    if (c.dataset.sampleId === sample.id) {
      c.classList.add("active-selected");
    } else {
      c.classList.remove("active-selected");
    }
  });

  btnAnalyze.disabled = false;
  btnAnalyze.classList.add("pulse-ready");
}

// ================= File Upload & Drag-and-Drop =================
function setupEventListeners() {
  // Benchmark switches
  bench10Btn.addEventListener("click", () => switchBenchmark("10class"));
  bench4Btn.addEventListener("click", () => switchBenchmark("4class"));

  // Drag and Drop
  dropZone.addEventListener("click", () => fileInput.click());

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
      handleUserFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleUserFile(e.target.files[0]);
    }
  });

  btnClearImg.addEventListener("click", resetImageInput);

  // Analyze Button
  btnAnalyze.addEventListener("click", executeScreening);

  // Grad-CAM opacity slider
  camOpacitySlider.addEventListener("input", (e) => {
    const val = e.target.value / 100;
    camImgOverlay.style.opacity = val;
  });

  // Export summary
  btnExportReport.addEventListener("click", () => {
    window.print();
  });
}

function switchBenchmark(bench) {
  if (activeBenchmark === bench) return;
  activeBenchmark = bench;

  if (bench === "10class") {
    bench10Btn.classList.add("active");
    bench4Btn.classList.remove("active");
    document.getElementById("sota-badge").innerHTML = `<span style="color: var(--accent-cyan); font-weight: 700;">10-Class SOTA: 91.83%</span>`;
  } else {
    bench4Btn.classList.add("active");
    bench10Btn.classList.remove("active");
    document.getElementById("sota-badge").innerHTML = `<span style="color: var(--accent-cyan); font-weight: 700;">4-Class SOTA: 95.74%</span>`;
  }

  populateModelOptions();
}

function handleUserFile(file) {
  if (!file.type.match("image.*")) {
    alert("Please upload a valid image file (JPG or PNG).");
    return;
  }

  currentFile = file;
  currentSampleId = null;

  document.querySelectorAll(".sample-card").forEach(c => {
    c.classList.remove("active-selected");
  });

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewBadge.textContent = `${file.name} (${Math.round(file.size / 1024)} KB)`;
    previewContainer.style.display = "flex";
    dropZone.style.display = "none";
    btnClearImg.style.display = "block";
    btnAnalyze.disabled = false;
    btnAnalyze.classList.add("pulse-ready");
  };
  reader.readAsDataURL(file);
}

function resetImageInput() {
  currentFile = null;
  currentSampleId = null;
  fileInput.value = "";
  previewContainer.style.display = "none";
  dropZone.style.display = "flex";
  btnClearImg.style.display = "none";
  btnAnalyze.disabled = true;
  btnAnalyze.classList.remove("pulse-ready");

  document.querySelectorAll(".sample-card").forEach(c => {
    c.classList.remove("active-selected");
  });

  resultsPlaceholder.style.display = "flex";
  resultsContainer.classList.remove("visible");
}

// ================= Clinical Inference Execution =================
async function executeScreening() {
  if (!currentFile && !currentSampleId) return;

  btnAnalyze.disabled = true;
  btnAnalyze.classList.remove("pulse-ready");
  btnAnalyzeText.innerHTML = `<span class="spinner"></span> Processing Screening & CAM...`;

  const formData = new FormData();
  formData.append("benchmark", activeBenchmark);
  formData.append("model_id", modelSelect.value);
  formData.append("generate_cam", chkGradcam.checked ? "true" : "false");

  if (currentFile) {
    formData.append("file", currentFile);
  } else if (currentSampleId) {
    formData.append("sample_id", currentSampleId);
  }

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Inference failed");
    }

    const data = await res.json();
    renderDiagnosticResults(data);
  } catch (err) {
    alert(`Clinical Screening Error: ${err.message}`);
    console.error(err);
  } finally {
    btnAnalyze.disabled = false;
    btnAnalyzeText.innerHTML = `Run Clinical Screening`;
  }
}

// ================= Render Diagnostic Results =================
function renderDiagnosticResults(data) {
  resultsPlaceholder.style.display = "none";
  resultsContainer.classList.add("visible");

  const top = data.top_prediction;
  const uncert = data.uncertainty;

  // Model & Prediction Badges
  resModelBadge.textContent = `${data.model_name} (Acc: ${data.model_accuracy})`;
  resTopDisease.textContent = top.short_name || top.class_name;
  resConfidenceNum.textContent = `${top.percentage}%`;

  // Urgency Pill Styling
  resUrgencyText.textContent = top.urgency || "Standard";
  resUrgencyPill.className = `urgency-pill urgency-${top.urgency_level || "blue"}`;
  if (top.urgency_level === "emerald") {
    resUrgencyIcon.textContent = "✓";
  } else if (top.urgency_level === "crimson") {
    resUrgencyIcon.textContent = "!";
  } else {
    resUrgencyIcon.textContent = "⚠";
  }

  // Certainty Pill
  resCertaintyPill.textContent = uncert.certainty_level;
  if (uncert.certainty_badge === "success") {
    resCertaintyPill.className = "urgency-pill urgency-emerald";
  } else if (uncert.certainty_badge === "warning") {
    resCertaintyPill.className = "urgency-pill urgency-amber";
  } else {
    resCertaintyPill.className = "urgency-pill urgency-red";
  }

  // Entropy & Consensus
  resEntropyVal.textContent = `${uncert.normalized_entropy} / 1.000 (Shannon: ${uncert.entropy})`;
  resConsensusVal.textContent = (data.model_id.includes("ensemble"))
    ? "Ensemble Consensus Aligned"
    : "Single High-Resolution Network";

  // Grad-CAM Heatmap & Overlay
  if (data.gradcam) {
    cardGradcam.style.display = "flex";
    camImgPreprocessed.src = data.gradcam.preprocessed_b64;
    camImgOverlay.src = data.gradcam.overlay_b64;
    camOpacitySlider.value = 100;
    camImgOverlay.style.opacity = "1.0";
  } else {
    cardGradcam.style.display = "none";
  }

  // Differential Diagnostic Spectrum
  resNumClasses.textContent = `${data.distribution.length} Target Conditions Analyzed`;
  probBarsContainer.innerHTML = "";

  data.distribution.forEach((item, index) => {
    const row = document.createElement("div");
    row.className = "prob-row";

    const isTop = index === 0;
    row.innerHTML = `
      <div class="prob-meta">
        <span class="prob-name" style="${isTop ? 'font-weight: 700; color: #fff;' : ''}">
          ${index + 1}. ${item.short_name || item.class_name}
        </span>
        <span class="prob-pct" style="${isTop ? 'color: var(--accent-emerald); font-weight: 800;' : ''}">
          ${item.percentage}%
        </span>
      </div>
      <div class="prob-track">
        <div class="prob-fill ${isTop ? 'top-rank' : ''}" style="width: 0%;"></div>
      </div>
    `;

    probBarsContainer.appendChild(row);

    // Trigger smooth fill animation
    setTimeout(() => {
      const fillEl = row.querySelector(".prob-fill");
      if (fillEl) {
        fillEl.style.width = `${Math.max(item.percentage, 0.8)}%`;
      }
    }, 40 + index * 25);
  });

  // Clinical Guidance & Management
  resDiseaseDesc.textContent = top.description;
  resRecommendationsList.innerHTML = "";
  if (top.recommendations && top.recommendations.length > 0) {
    top.recommendations.forEach(rec => {
      const li = document.createElement("li");
      li.textContent = rec;
      resRecommendationsList.appendChild(li);
    });
  }

  // Scroll smoothly to results on small screens
  if (window.innerWidth < 1180) {
    resultsContainer.scrollIntoView({ behavior: "smooth" });
  }
}
