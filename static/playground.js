// ────────────────────────────────────────────────
// Default system prompt
const DEFAULT_SYSTEM_PROMPT = `You are a helpful AI assistant that answers questions based on the provided context documents.

Key guidelines:
- Answer directly and concisely
- Cite specific documents when possible
- If the context doesn't contain the answer, say so
- Be honest about uncertainty`.trim();

// State
let systemPrompt = localStorage.getItem('ragSystemPrompt') || DEFAULT_SYSTEM_PROMPT;
let lastQuestion = "";
let lastContextDocs = [];
let lastPayload = null;

// Init DOM
const systemPromptEl = document.getElementById('systemPrompt');
systemPromptEl.value = systemPrompt;

// Buttons
document.getElementById("btnGenerate").addEventListener("click", generateAnswer);
document.getElementById("btnPreview").addEventListener("click", previewFullPrompt);
document.getElementById("btnReset").addEventListener("click", resetSystemPrompt);
document.getElementById("btnCloseModal").addEventListener("click", closeModal);

// Slider value displays
document.getElementById('nContextDocs').addEventListener('input', e => {
  document.getElementById('nDocsValue').textContent = e.target.value;
  updateTokenEstimate();
});

document.getElementById('temperature').addEventListener('input', e => {
  document.getElementById('tempValue').textContent = parseFloat(e.target.value).toFixed(1);
});

// Save prompt whenever it changes
systemPromptEl.addEventListener('input', e => {
  systemPrompt = e.target.value.trim();
  localStorage.setItem('ragSystemPrompt', systemPrompt);
  updateTokenEstimate();
});

// Update estimate when question/toggles change
document.getElementById("questionInput").addEventListener("input", updateTokenEstimate);
document.querySelectorAll('input[name="promptMode"]').forEach(r => r.addEventListener("change", updateTokenEstimate));
document.getElementById("streamPreview").addEventListener("change", updateTokenEstimate);

// Modal click outside to close
document.getElementById('previewModal').addEventListener('click', e => {
  if (e.target === document.getElementById('previewModal')) closeModal();
});

// ────────────────────────────────────────────────
function resetSystemPrompt() {
  if (confirm("Reset system prompt to default? This cannot be undone.")) {
    systemPrompt = DEFAULT_SYSTEM_PROMPT;
    systemPromptEl.value = systemPrompt;
    localStorage.setItem('ragSystemPrompt', systemPrompt);
    updateTokenEstimate();
  }
}

function closeModal() {
  document.getElementById('previewModal').style.display = 'none';
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text ?? "";
  return div.innerHTML;
}

function getPromptMode() {
  const selected = document.querySelector('input[name="promptMode"]:checked');
  return selected ? selected.value : "actual";
}

// Rough token estimate: ~ chars/4
function estimateTokens(text) {
  if (!text) return 0;
  return Math.max(1, Math.round(text.length / 4));
}

function buildSimulatedContext(nDocs) {
  const k = Math.max(1, Math.min(10, parseInt(nDocs, 10)));
  return Array(k)
    .fill("[Retrieved Document X]\nContent...\n")
    .map((v, i) => v.replace("X", i + 1))
    .join("\n");
}

function formatActualContextDocs(docs) {
  if (!docs || docs.length === 0) return "(No documents retrieved yet)";

  return docs.map((doc, i) => {
    const docId =
      doc.source ||
      doc.doc_id ||
      doc.id ||
      (doc.metadata && (doc.metadata.filename || doc.metadata.doc_id)) ||
      `Doc ${i + 1}`;

    const text = doc.text || doc.content || "";
    return `[${i + 1}] ${docId}\n${text}`.trim();
  }).join("\n\n");
}

function buildFinalPrompt(question, contextText) {
  return `Context information from relevant documents:

${contextText}

Based on the context above, please answer the following question.
If the context doesn't contain enough information, say so.

Question: ${question}

Answer:`;
}

function updateTokenEstimate() {
  const mode = getPromptMode();
  const question = document.getElementById('questionInput').value.trim();
  const nDocs = document.getElementById('nContextDocs').value;

  const contextText =
    mode === "actual" ? formatActualContextDocs(lastContextDocs) : buildSimulatedContext(nDocs);

  const fullPrompt = buildFinalPrompt(question || "[question]", contextText);
  const tokens = estimateTokens(`${systemPrompt}\n\n${fullPrompt}`);
  document.getElementById("tokenEstimate").textContent = `Token estimate: ~${tokens} tokens`;
}

// Typewriter streaming preview
async function streamText(el, text) {
  el.innerHTML = "";
  const safe = escapeHtml(text || "");
  const step = 4;
  for (let i = 0; i <= safe.length; i += step) {
    el.innerHTML = `<div style="white-space:pre-wrap;">${safe.slice(0, i)}</div>`;
    await new Promise(r => setTimeout(r, 10));
  }
}

// ────────────────────────────────────────────────
async function generateAnswer() {
  const question = document.getElementById('questionInput').value.trim();
  if (!question) {
    alert("Please enter a question first.");
    return;
  }

  const nDocs = parseInt(document.getElementById('nContextDocs').value, 10);
  const temp = parseFloat(document.getElementById('temperature').value);

  const responseArea = document.getElementById('responseArea');
  const sourcesSection = document.getElementById('sourcesSection');
  const metadataDiv = document.getElementById('metadata');

  responseArea.innerHTML = '<div class="loading-state">Generating answer...</div>';
  sourcesSection.innerHTML = '';
  metadataDiv.innerHTML = '';

  try {
    const startTime = performance.now();

    const payload = {
      question,
      n_context_docs: nDocs,
      temperature: temp,
      system_prompt: systemPrompt
    };
    lastPayload = payload;

    const res = await fetch('http://localhost:8000/rag', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const duration = ((performance.now() - startTime) / 1000).toFixed(2);
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || 'API error');
    }

    // Store real context for "Actual" prompt preview
    lastQuestion = question;
    lastContextDocs = data.context || [];
    updateTokenEstimate();

    // Answer rendering
    if (document.getElementById("streamPreview").checked) {
      await streamText(responseArea, data.answer || "");
    } else {
      responseArea.innerHTML = `<div style="white-space:pre-wrap;">${escapeHtml(data.answer || "")}</div>`;
    }

    metadataDiv.innerHTML = `
      <strong>Response time:</strong> ${duration}s •
      <strong>Context docs used:</strong> ${data.context_count}
    `;

    // Render sources (safe + compatible with your backend shape)
    if (data.context?.length > 0) {
      let html = '';
      data.context.forEach((src, i) => {
        const sourceName =
          src.source ||
          src.doc_id ||
          src.id ||
          (src.metadata && (src.metadata.filename || src.metadata.doc_id)) ||
          `Doc ${i + 1}`;

        html += `
          <div class="source-card">
            <div class="source-header">[Doc ${i+1}] ${escapeHtml(sourceName)}</div>
            ${src.page ? `<div class="metadata-line">Page: ${escapeHtml(String(src.page))}</div>` : ''}
            ${(src.score !== undefined && src.score !== null)
              ? `<div class="metadata-line">Relevance score: ${escapeHtml(String(src.score))}</div>`
              : ''
            }
          </div>
        `;
      });
      sourcesSection.innerHTML = html;
    } else {
      sourcesSection.innerHTML = '<p style="color:#9ca3af;">No sources retrieved.</p>';
    }

  } catch (err) {
    responseArea.innerHTML = `<p class="error">Error: ${escapeHtml(err.message || String(err))}</p>`;
  }
}

// ────────────────────────────────────────────────
function previewFullPrompt() {
  const mode = getPromptMode();

  const typedQuestion = document.getElementById('questionInput').value.trim();
  const question = (mode === "actual" ? lastQuestion : typedQuestion) || "[Your question will appear here]";
  const nDocs = document.getElementById('nContextDocs').value;

  const contextText =
    (mode === "actual") ? formatActualContextDocs(lastContextDocs) : buildSimulatedContext(nDocs);

  const fullPrompt = buildFinalPrompt(question, contextText);
  const tokens = estimateTokens(`${systemPrompt}\n\n${fullPrompt}`);

  const rawPayload = lastPayload ? JSON.stringify(lastPayload, null, 2) : "(no /rag call yet)";

  document.getElementById('previewContent').textContent =
    `=== System Prompt ===\n${systemPrompt}\n\n` +
    `=== Context Documents (${mode}) ===\n${contextText}\n\n` +
    `=== Final Prompt Sent to LLM ===\n${fullPrompt}\n\n` +
    `=== Raw JSON Payload ===\n${rawPayload}\n\n` +
    `=== Token Estimate ===\n~${tokens} tokens`;

  document.getElementById('previewModal').style.display = 'flex';

  // also update the small token line on the left
  document.getElementById("tokenEstimate").textContent = `Token estimate: ~${tokens} tokens`;
}

// initial estimate
updateTokenEstimate();