// PageCiteRAG browser reader. Talks only to the same-origin local API with the bearer token;
// the token lives in sessionStorage for this tab. All document text is inserted as text, never
// as HTML, so quotations and answers cannot inject markup.
"use strict";

const TOKEN_KEY = "pagecite-token";
const $ = (id) => document.getElementById(id);
const state = { token: null, sources: [], quotes: new Map(), details: new Map(), pageCache: new Map(), active: null };

function readToken() {
  try { return sessionStorage.getItem(TOKEN_KEY); } catch (e) { return null; }
}
function saveToken(value) {
  try { if (value) sessionStorage.setItem(TOKEN_KEY, value); else sessionStorage.removeItem(TOKEN_KEY); } catch (e) { /* tab-only */ }
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { ...(options.headers || {}), Authorization: `Bearer ${state.token}` },
  });
  if (response.status === 401) {
    showTokenForm("The token was not accepted.");
    throw new Error("unauthorized");
  }
  return response;
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else node.setAttribute(key, value);
  }
  for (const child of children) node.append(child);
  return node;
}

// ---------- token and document ----------
function showTokenForm(message = "") {
  $("app").hidden = true;
  $("forget-token").hidden = true;
  $("token-form").hidden = false;
  $("token-error").textContent = message;
  $("token-input").focus();
}

async function connect() {
  try {
    const response = await api("/v1/document");
    if (!response.ok) throw new Error(`document ${response.status}`);
    const info = (await response.json()).document;
    const pages = info.page_count ? ` · ${info.page_count} pages` : "";
    $("doc-title").textContent = `${info.source_name}${pages}`;
    $("token-form").hidden = true;
    $("app").hidden = false;
    $("forget-token").hidden = false;
    $("question").focus();
  } catch (error) {
    if (error.message !== "unauthorized") showTokenForm("Could not reach the local service.");
  }
}

// ---------- answer rendering ----------
function inline(text, parent) {
  // **bold** and [n] source markers; everything else stays plain text.
  const pattern = /\*\*([^*]+)\*\*|\[(\d{1,3})\]/g;
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    if (match.index > last) parent.append(text.slice(last, match.index));
    if (match[1] !== undefined) parent.append(el("strong", { text: match[1] }));
    else parent.append(citeButton(Number(match[2])));
    last = match.index + match[0].length;
  }
  if (last < text.length) parent.append(text.slice(last));
}

function citeButton(number) {
  const source = state.sources[number - 1];
  if (!source) return document.createTextNode(`[${number}]`);
  const button = el("button", { type: "button", class: "cite", "data-n": String(number), "aria-pressed": "false", "aria-label": `Source ${number}` });
  button.textContent = String(number);
  button.addEventListener("click", () => showSource(number));
  button.addEventListener("mouseenter", () => showTip(button, number));
  button.addEventListener("focus", () => showTip(button, number));
  button.addEventListener("mouseleave", hideTip);
  button.addEventListener("blur", hideTip);
  return button;
}

function renderText(text, container) {
  container.replaceChildren();
  let list = null;
  for (const raw of text.split(/\n/)) {
    const line = raw.trim();
    const bullet = /^(?:[*•-]|\d+\.)\s+(.*)$/.exec(line);
    if (bullet) {
      if (!list) { list = el(/^\d/.test(line) ? "ol" : "ul"); container.append(list); }
      const item = el("li");
      inline(bullet[1], item);
      list.append(item);
      continue;
    }
    list = null;
    if (!line) continue;
    const paragraph = el("p");
    inline(line.replace(/^#+\s*/, ""), paragraph);
    container.append(paragraph);
  }
}

function collectSources(result) {
  // Quote-first answers number their sources in explanation_sources (as the CLI prints them);
  // claim-only answers number the claims' citations in order.
  state.quotes = new Map();
  for (const claim of [...result.answer_claims, ...result.detail_claims]) {
    for (const citation of claim.citations) {
      if (!state.quotes.has(citation.block_id)) state.quotes.set(citation.block_id, citation);
    }
  }
  if (result.explanation) return [...result.explanation_sources];
  return [...state.quotes.keys()];
}

function renderAnswer(payload) {
  const { result, trace } = payload;
  state.sources = collectSources(result);
  state.details = new Map();
  state.active = null;
  const meta = $("answer-meta");
  meta.replaceChildren();
  const status = { answered: "answered", partial: "partial answer", insufficient_evidence: "not enough evidence" }[result.status] || result.status;
  meta.append(el("span", { class: result.status === "answered" ? "pill" : "pill warn", text: status }));
  meta.append(el("span", { text: `mode ${trace.mode || result.mode}` }));
  meta.append(el("span", { text: `${(trace.total_ms / 1000).toFixed(0)} s` }));
  meta.append(el("span", { text: `${state.sources.length} sources` }));

  if (result.explanation) {
    renderText(result.explanation, $("answer-text"));
  } else if (result.answer_claims.length) {
    const text = result.answer_claims.map((claim) => {
      const numbers = claim.citations.map((c) => state.sources.indexOf(c.block_id) + 1).filter((n) => n > 0);
      return `- ${claim.text} ${numbers.map((n) => `[${n}]`).join("")}`;
    }).join("\n");
    renderText(text, $("answer-text"));
  } else {
    renderText(result.answer, $("answer-text"));
  }

  const list = $("sources");
  list.replaceChildren();
  state.sources.forEach((blockId, index) => {
    const citation = state.quotes.get(blockId);
    const page = citation && citation.page_number ? `page ${citation.page_number}` : "source";
    const body = el("div", {}, el("div", { class: "where", text: page }), el("div", { class: "q", text: citation ? citation.quote : "Open to read this passage." }));
    list.append(el("li", {}, citeButton(index + 1), body));
  });
  $("sources-title").hidden = state.sources.length === 0;
  $("answer-note").textContent = result.limitations.join(" ");
  $("answer").hidden = false;
  resetViewer();
}

// ---------- tooltip ----------
function showTip(button, number) {
  const citation = state.quotes.get(state.sources[number - 1]);
  const tip = $("tip");
  tip.replaceChildren(el("b", { text: citation && citation.page_number ? `Source ${number} · page ${citation.page_number}` : `Source ${number}` }));
  tip.append(citation ? clip(citation.quote, 320) : "Click to read this passage.");
  tip.hidden = false;
  const box = button.getBoundingClientRect();
  const width = tip.offsetWidth;
  tip.style.left = `${Math.max(12, Math.min(box.left, window.innerWidth - width - 12))}px`;
  const above = box.top - tip.offsetHeight - 8;
  tip.style.top = `${above > 8 ? above : box.bottom + 8}px`;
}
function hideTip() { $("tip").hidden = true; }
function clip(text, limit) { return text.length > limit ? `${text.slice(0, limit - 1)}…` : text; }

// ---------- source viewer ----------
function resetViewer() {
  $("viewer-empty").hidden = false;
  $("viewer-body").hidden = true;
}

async function sourceDetails(blockId) {
  if (!state.details.has(blockId)) {
    const response = await api(`/v1/source/${blockId}`);
    if (!response.ok) throw new Error(`source ${response.status}`);
    state.details.set(blockId, await response.json());
  }
  return state.details.get(blockId);
}

async function pageImage(pageNumber) {
  if (!state.pageCache.has(pageNumber)) {
    const response = await api(`/v1/pages/${pageNumber}`);
    if (!response.ok) return null;
    const blob = await response.blob();
    state.pageCache.set(pageNumber, {
      url: URL.createObjectURL(blob),
      width: Number(response.headers.get("X-Page-Width-Points")),
      height: Number(response.headers.get("X-Page-Height-Points")),
    });
  }
  return state.pageCache.get(pageNumber);
}

function regionsOn(details, pageNumber) {
  const location = details.native_location;
  const regions = location && Array.isArray(location.regions) ? location.regions : [];
  return regions.filter((region) => region.page_number === pageNumber && region.bbox);
}

async function showSource(number) {
  hideTip();
  state.active = number;
  document.querySelectorAll(".cite").forEach((button) => button.setAttribute("aria-pressed", String(Number(button.dataset.n) === number)));
  const blockId = state.sources[number - 1];
  const citation = state.quotes.get(blockId);
  $("viewer-empty").hidden = true;
  $("viewer-body").hidden = false;
  $("viewer-badge").textContent = String(number);
  $("viewer-quote").textContent = citation ? citation.quote : "";
  $("viewer-page").replaceChildren();
  $("viewer-hint").textContent = "Loading the page…";
  try {
    const details = await sourceDetails(blockId);
    if (state.active !== number) return;
    if (!citation) $("viewer-quote").textContent = details.text;
    const pageNumber = (citation && citation.page_number) || details.page_number;
    $("viewer-where").textContent = pageNumber ? `Page ${pageNumber}` : "Source passage";
    const image = pageNumber ? await pageImage(pageNumber) : null;
    if (state.active !== number) return;
    if (!image) {
      $("viewer-hint").textContent = "Start serve with --pdf to see the page itself.";
      return;
    }
    const frame = $("viewer-page");
    frame.append(el("img", { src: image.url, alt: `Page ${pageNumber}` }));
    let first = null;
    for (const region of regionsOn(details, pageNumber)) {
      const { l, r, t, b } = region.bbox;
      const top = region.bbox.coord_origin === "TOPLEFT" ? t : image.height - t;
      const height = Math.abs(t - b);
      const box = el("div", { class: "box" });
      box.style.left = `${(l / image.width) * 100}%`;
      box.style.width = `${((r - l) / image.width) * 100}%`;
      box.style.top = `${(top / image.height) * 100}%`;
      box.style.height = `${(height / image.height) * 100}%`;
      frame.append(box);
      first = first || box;
    }
    $("viewer-hint").textContent = first ? "The highlighted box is where the quoted passage sits on the page." : "This passage has no recorded position on the page.";
    if (first) requestAnimationFrame(() => first.scrollIntoView({ block: "center", behavior: "smooth" }));
  } catch (error) {
    if (error.message !== "unauthorized") $("viewer-hint").textContent = "Could not load this source.";
  }
}

// ---------- asking ----------
async function ask(event) {
  event.preventDefault();
  const question = $("question").value.trim();
  if (!question) return;
  const button = $("ask-button");
  button.disabled = true;
  const started = Date.now();
  const tick = () => { $("ask-status").textContent = `Reading the document… ${Math.round((Date.now() - started) / 1000)} s`; };
  tick();
  const timer = setInterval(tick, 1000);
  try {
    const body = { question };
    if ($("mode").value) body.mode = $("mode").value;
    const response = await api("/v1/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      $("ask-status").textContent = `Failed: ${typeof detail.detail === "string" ? detail.detail : response.status}`;
      return;
    }
    renderAnswer(await response.json());
    $("ask-status").textContent = "";
  } catch (error) {
    if (error.message !== "unauthorized") $("ask-status").textContent = "Could not reach the local service.";
  } finally {
    clearInterval(timer);
    button.disabled = false;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  $("token-form").addEventListener("submit", (event) => {
    event.preventDefault();
    state.token = $("token-input").value.trim();
    saveToken(state.token);
    $("token-input").value = "";
    connect();
  });
  $("forget-token").addEventListener("click", () => { saveToken(null); state.token = null; showTokenForm(); });
  $("ask-form").addEventListener("submit", ask);
  $("question").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) $("ask-form").requestSubmit();
  });
  state.token = readToken();
  if (state.token) connect(); else showTokenForm();
});
