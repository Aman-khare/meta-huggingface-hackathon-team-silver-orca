// Clinical Note Scribe — Frontend Logic

const API = "";

// DOM refs
const taskSelect = document.getElementById("taskSelect");
const resetBtn = document.getElementById("resetBtn");
const stepBtn = document.getElementById("stepBtn");
const statusBadge = document.getElementById("statusBadge");
const contextSection = document.getElementById("contextSection");
const contextGrid = document.getElementById("contextGrid");
const transcriptArea = document.getElementById("transcriptArea");
const actionSection = document.getElementById("actionSection");
const actionType = document.getElementById("actionType");
const sectionSelect = document.getElementById("sectionSelect");
const soapInputs = document.getElementById("soapInputs");
const reviseInput = document.getElementById("reviseInput");
const clarifyInput = document.getElementById("clarifyInput");
const rewardSection = document.getElementById("rewardSection");
const scoreValue = document.getElementById("scoreValue");
const rewardFill = document.getElementById("rewardFill");
const draftArea = document.getElementById("draftArea");
const draftEmpty = document.getElementById("draftEmpty");
const soapDraft = document.getElementById("soapDraft");
const soapGrid = document.getElementById("soapGrid");
const logContainer = document.getElementById("logContainer");

let currentObs = null;
let isDone = false;

// Logging
function addLog(msg, type = "") {
  const time = new Date().toLocaleTimeString("en-US", { hour12: false });
  const entry = document.createElement("div");
  entry.className = "log-entry " + type;
  entry.innerHTML = `<span class="log-time">${time}</span>${msg}`;
  logContainer.prepend(entry);
}

// Status badge
function setStatus(state) {
  statusBadge.className = "status-badge " + state;
  statusBadge.textContent = state === "idle" ? "Idle" : state === "active" ? "Active" : "Done";
}

// Toggle action inputs based on action type
actionType.addEventListener("change", () => {
  const val = actionType.value;
  soapInputs.style.display = val === "submit_note" ? "block" : "none";
  reviseInput.style.display = val === "revise_section" ? "block" : "none";
  clarifyInput.style.display = val === "request_clarify" ? "block" : "none";
  sectionSelect.style.display = val === "revise_section" ? "inline-block" : "none";
});

// Format transcript with speaker highlighting
function renderTranscript(text) {
  if (!text) return "";
  const lines = text.split("\n");
  return lines.map(line => {
    if (/^(Dr\.|Doctor)/i.test(line.trim())) {
      return `<div><span class="speaker-doctor">${escapeHtml(line)}</span></div>`;
    } else if (/^(Patient|Pt)/i.test(line.trim())) {
      return `<div><span class="speaker-patient">${escapeHtml(line)}</span></div>`;
    }
    return `<div>${escapeHtml(line)}</div>`;
  }).join("");
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// Render patient context as cards
function renderContext(ctx) {
  if (!ctx || Object.keys(ctx).length === 0) {
    contextSection.style.display = "none";
    return;
  }
  contextSection.style.display = "block";
  contextGrid.innerHTML = "";

  const flat = flattenContext(ctx);
  for (const [key, val] of Object.entries(flat)) {
    const card = document.createElement("div");
    card.className = "context-card";
    card.innerHTML = `<div class="label">${escapeHtml(key)}</div><div class="value">${escapeHtml(String(val))}</div>`;
    contextGrid.appendChild(card);
  }
}

function flattenContext(obj, prefix = "") {
  const result = {};
  for (const [k, v] of Object.entries(obj)) {
    const key = prefix ? `${prefix} › ${k}` : k;
    if (v && typeof v === "object" && !Array.isArray(v)) {
      Object.assign(result, flattenContext(v, key));
    } else if (Array.isArray(v)) {
      result[key] = v.length > 0 ? v.join(", ") : "—";
    } else {
      result[key] = v ?? "—";
    }
  }
  return result;
}

// Render SOAP draft
function renderDraft(draftText) {
  if (!draftText) {
    draftEmpty.style.display = "flex";
    soapDraft.style.display = "none";
    return;
  }
  draftEmpty.style.display = "none";
  soapDraft.style.display = "block";

  const sections = { S: "", O: "", A: "", P: "" };
  const lines = draftText.split("\n");
  for (const line of lines) {
    for (const prefix of ["S: ", "O: ", "A: ", "P: "]) {
      if (line.startsWith(prefix)) {
        sections[prefix[0]] = line.slice(prefix.length);
      }
    }
  }

  const labels = { S: "Subjective", O: "Objective", A: "Assessment", P: "Plan" };
  soapGrid.innerHTML = "";
  for (const [key, label] of Object.entries(labels)) {
    const card = document.createElement("div");
    card.className = `soap-card ${key.toLowerCase()}`;
    card.innerHTML = `
      <div class="soap-label">${label}</div>
      <div class="soap-text">${escapeHtml(sections[key]) || '<em style="opacity:0.4">Empty</em>'}</div>
    `;
    soapGrid.appendChild(card);
  }
}

// Render reward
function renderReward(rewardObj) {
  if (!rewardObj) {
    rewardSection.style.display = "none";
    return;
  }
  rewardSection.style.display = "block";
  const val = rewardObj.value;
  scoreValue.textContent = val.toFixed(4);
  rewardFill.style.width = (val * 100) + "%";

  if (val >= 0.7) {
    scoreValue.style.color = "var(--green)";
  } else if (val >= 0.4) {
    scoreValue.style.color = "var(--yellow)";
  } else {
    scoreValue.style.color = "var(--red)";
  }
}

// Update UI from observation
function updateUI(obs, reward = null, done = false) {
  currentObs = obs;
  isDone = done;

  transcriptArea.innerHTML = `<div class="transcript-box">${renderTranscript(obs.transcript)}</div>`;
  renderContext(obs.patient_context);
  renderDraft(obs.current_draft);

  if (reward) renderReward(reward);

  actionSection.style.display = done ? "none" : "block";
  setStatus(done ? "done" : "active");

  if (done) {
    addLog(`Episode complete — score: ${reward ? reward.value.toFixed(4) : "N/A"}`, "success");
  }
}

// Reset
resetBtn.addEventListener("click", async () => {
  const taskId = taskSelect.value;
  resetBtn.disabled = true;
  addLog(`Resetting with task: ${taskId}`);

  try {
    const res = await fetch(`${API}/reset`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_id: taskId }),
    });
    if (!res.ok) throw new Error(await res.text());
    const obs = await res.json();
    rewardSection.style.display = "none";
    updateUI(obs);
    addLog("Environment reset successfully", "success");
  } catch (err) {
    addLog(`Reset failed: ${err.message}`, "error");
  } finally {
    resetBtn.disabled = false;
  }
});

// Step
stepBtn.addEventListener("click", async () => {
  if (isDone) {
    addLog("Episode is done. Reset first.", "error");
    return;
  }

  const action = actionType.value;
  let payload = {};

  if (action === "submit_note") {
    payload = {
      action_type: "submit_note",
      soap_note: {
        subjective: document.getElementById("inputS").value,
        objective: document.getElementById("inputO").value,
        assessment: document.getElementById("inputA").value,
        plan: document.getElementById("inputP").value,
      },
    };
  } else if (action === "revise_section") {
    payload = {
      action_type: "revise_section",
      section: sectionSelect.value,
      revision_text: document.getElementById("inputRevision").value,
    };
  } else if (action === "request_clarify") {
    payload = {
      action_type: "request_clarify",
      clarify_question: document.getElementById("inputClarify").value,
    };
  }

  stepBtn.disabled = true;
  addLog(`Sending action: ${action}`);

  try {
    const res = await fetch(`${API}/step`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    updateUI(data.observation, data.reward, data.done);

    if (data.info && data.info.clarify_answer) {
      addLog(`Clarify answer: ${data.info.clarify_answer}`);
    }
    addLog(`Step done — reward: ${data.reward.value.toFixed(4)}, done: ${data.done}`);
  } catch (err) {
    addLog(`Step failed: ${err.message}`, "error");
  } finally {
    stepBtn.disabled = false;
  }
});

// Init
addLog("Frontend loaded. Select a task and click Reset.");
