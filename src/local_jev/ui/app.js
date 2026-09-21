/* local-jev project UI. No build step, no dependencies. */
(() => {
  "use strict";
  const $ = (sel, root = document) => root.querySelector(sel);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (n) => Number(n || 0).toLocaleString();
  const TYPE_LABEL = { noul: "Yes / No", choice: "Category", score: "Score" };
  const LIMITS = [0, 50, 100, 500, 1000, 5000];
  const PAGE = 50;

  const S = { projects: [], pid: null, project: null, results: null, items: [], total: 0, filters: [], search: "",
              model: null, models: [], limit: 0, expanded: new Set(), showAll: new Set(), openItem: null, poll: null, lastRun: null,
              view: "test", drafts: {}, askMore: new Set(), explain: false };

  const store = { get: (k, d) => { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } },
                  set: (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} } };

  async function api(path, opts = {}) {
    const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts,
                                    body: opts.body ? JSON.stringify(opts.body) : undefined });
    if (!res.ok) {
      let msg = res.statusText;
      try { const d = (await res.json()).detail; msg = typeof d === "string" ? d : (d || []).map((e) => e.msg).join("; ") || msg; } catch {}
      throw new Error(msg);
    }
    return res.status === 204 ? null : res.json();
  }

  let toastTimer;
  function toast(msg) {
    const el = $("#toast");
    el.textContent = msg; el.classList.remove("hidden");
    clearTimeout(toastTimer); toastTimer = setTimeout(() => el.classList.add("hidden"), 3200);
  }

  const noun = (n) => {
    const plural = S.project?.noun || "items";
    return n === 1 ? plural.replace(/ies$/, "y").replace(/s$/, "") : plural;
  };
  const qs = () => new URLSearchParams({ model: S.model || "", limit: S.limit || 0 });

  /* ---------- loading ---------- */
  async function boot() {
    const meta = await api("/api/meta");
    S.models = meta.models;
    S.model = store.get("model", null);
    if (!S.models.some((m) => m.name === S.model && !m.unavailable)) {   // saved model gone or unloadable: use the server's default
      if (S.model) store.set("model", null);
      S.model = meta.default_model;
    }
    renderPickers();
    $("#engine-status").title = `local-jev ${meta.version} · default model ${meta.default_model}`;
    $("#limit").innerHTML = LIMITS.map((n) => `<option value="${n}">${n ? "Latest " + fmt(n) : "All items"}</option>`).join("");
    await loadProjects(store.get("pid", null));
  }

  async function loadProjects(select) {
    S.projects = await api("/api/projects");
    const has = S.projects.length > 0;
    $("#main").classList.toggle("hidden", !has);
    $(".sidebar").classList.toggle("hidden", !has);
    $("#welcome").classList.toggle("hidden", has);
    if (!has) { S.pid = null; renderTabs(); return; }
    const pid = S.projects.some((p) => p.id === select) ? select : S.projects[0].id;
    await openProject(pid);
  }

  async function openProject(pid) {
    if (S.page === "models") leaveModels();
    S.pid = pid; store.set("pid", pid);
    S.filters = []; S.search = ""; $("#search").value = ""; S.expanded.clear(); S.openItem = null;
    S.limit = store.get("limit:" + pid, 0); $("#limit").value = String(S.limit);
    S.view = store.get("view:" + pid, "test") === "batch" ? "batch" : "test";
    S.askMore.clear();
    await refresh();
  }

  async function refresh({ keepItems = false } = {}) {
    S.project = await api(`/api/projects/${S.pid}`);
    const me = S.projects.find((p) => p.id === S.pid);
    if (me) { me.name = S.project.name; me.item_count = S.project.item_count; }
    renderTabs(); renderHead(); renderQuestions(); renderView();
    await loadResults();
    if (!keepItems) await loadItems(true);
    renderAsk();
  }

  async function loadResults() {
    S.results = await api(`/api/projects/${S.pid}/results?${qs()}`);
    renderResults(); renderRunbar();
    const running = S.results.run?.state === "running";
    if (running && !S.poll) S.poll = setInterval(tick, 900);
    if (!running && S.poll) { clearInterval(S.poll); S.poll = null; }
  }

  async function tick() {
    const wasRunning = S.results?.run?.state === "running";
    await loadResults();
    if (wasRunning && S.results.run.state !== "running") {
      await loadItems(true);
      if (S.results.run.state === "error") toast("Run failed: " + S.results.run.error);
    }
  }

  async function loadItems(reset) {
    if (reset) { S.items = []; }
    const p = qs(); p.set("filters", JSON.stringify(S.filters)); p.set("offset", S.items.length); p.set("page", PAGE); p.set("search", S.search);
    const data = await api(`/api/projects/${S.pid}/items?${p}`);
    S.items = S.items.concat(data.items); S.total = data.total;
    renderItems();
  }

  /* ---------- rendering ---------- */
  function renderTabs() {
    $("#tabs").innerHTML = S.projects.map((p) =>
      `<button class="tab ${p.id === S.pid ? "active" : ""}" data-pid="${p.id}">${esc(p.name)} <span class="count">${fmt(p.item_count)}</span></button>`).join("");
  }

  function renderHead() {
    const p = S.project;
    $("#project-name").textContent = p.name;
    $("#project-desc").textContent = p.description || "";
    $("#item-count").textContent = `${fmt(p.item_count)} ${noun(p.item_count)}`;
    $("#item-updated").textContent = p.items_updated_at ? "Added " + when(p.items_updated_at) : "No data yet";
    $("#search").placeholder = `Search ${p.noun || "items"}…`;
  }

  function when(ts) {
    const d = new Date(ts * 1000), now = new Date();
    const time = d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    if (d.toDateString() === now.toDateString()) return "today at " + time;
    return d.toLocaleDateString([], { month: "short", day: "numeric" }) + " at " + time;
  }

  function renderQuestions() {
    const list = S.project.questions;
    $("#questions-empty").classList.toggle("hidden", list.length > 0);
    $("#questions").innerHTML = list.map((q) => `
      <li data-qid="${q.id}">
        <input type="checkbox" ${q.enabled ? "checked" : ""} title="Include in the next run" aria-label="Include ${esc(q.name)} in the next run">
        <div class="q-body"><div class="q-name">${esc(q.name)}</div><div class="q-type">${TYPE_LABEL[q.type]}</div></div>
        <button class="q-edit" title="Edit question"><svg class="ico" viewBox="0 0 24 24"><path d="M4 20l1-4L16.5 4.5a2.1 2.1 0 013 3L8 19l-4 1z"/></svg></button>
      </li>`).join("");
  }

  function renderRunbar() {
    const r = S.results, run = r.run || {}, btn = $("#run"), label = $("span", btn);
    const progress = $("#run-progress");
    btn.className = "btn btn-run"; btn.disabled = false; label.textContent = "Run";
    progress.classList.add("hidden");
    const enabled = S.project.questions.filter((q) => q.enabled).length;
    if (run.state === "running") {
      const rate = run.done / Math.max(run.elapsed, 0.1);
      $("#run-title").textContent = `Sorting… ${fmt(run.done)} of ${fmt(run.total)} ${noun(run.total)}`;
      $("#run-sub").textContent = `${run.questions} question${run.questions === 1 ? "" : "s"} each · ${rate.toFixed(1)}/sec` +
        (rate > 0 ? ` · about ${eta((run.total - run.done) / rate)} left` : "");
      progress.classList.remove("hidden"); $("div", progress).style.width = (100 * run.done / Math.max(run.total, 1)) + "%";
      btn.classList.add("stop"); label.textContent = "Stop"; return;
    }
    if (!S.project.item_count) { $("#run-title").textContent = "Add some data to get started."; $("#run-sub").textContent = "CSV, JSON, text, .eml and .mbox files all work."; btn.disabled = true; return; }
    if (!S.project.questions.length) { $("#run-title").textContent = "Add a question to get started."; $("#run-sub").textContent = ""; btn.disabled = true; return; }
    if (!enabled) { $("#run-title").textContent = "Tick at least one question to run."; $("#run-sub").textContent = ""; btn.disabled = true; return; }
    const pending = $("#scope").value === "all" ? r.items : r.unsorted;
    if (pending > 0) {
      $("#run-title").textContent = $("#scope").value === "all" ? `Re-sort all ${fmt(r.items)} ${noun(r.items)} from scratch.` : `${fmt(pending)} ${noun(pending)} not sorted yet.`;
      btn.classList.add("ready");
    } else {
      $("#run-title").textContent = `All caught up. Select any result below to filter the ${S.project.noun || "items"}.`;
      btn.disabled = true;
    }
    const done = run.state === "done" || run.state === "cancelled";
    $("#run-sub").innerHTML = done
      ? `<span class="okmark">✓</span> Sorted ${fmt(run.done)} in ${eta(run.elapsed)} on this machine, for $0.00.` +
        (run.truncated ? ` ${fmt(run.truncated)} long ${noun(run.truncated)} shortened to fit.` : "") + (run.state === "cancelled" ? " Stopped early." : "")
      : run.state === "error" ? `<span style="color:var(--danger)">Last run failed: ${esc(run.error)}</span>` : "";
  }

  const eta = (sec) => sec < 90 ? `${Math.max(1, Math.round(sec))} sec` : sec < 5400 ? `${Math.round(sec / 60)} min` : `${(sec / 3600).toFixed(1)} hr`;

  function renderResults() {
    const r = S.results, host = $("#results");
    if (!r.questions.length) { host.innerHTML = `<div class="grid-empty">Results appear here once you add a question and press Run.</div>`; return; }
    host.innerHTML = r.questions.map((q) => {
      const active = S.filters.find((f) => f.q === q.question_id);
      const max = Math.max(1, ...q.rows.map((x) => x.count));
      const all = S.showAll.has(q.question_id) || q.rows.length <= 5;
      const rows = (all ? q.rows : q.rows.slice(0, 5)).map((x) => `
        <div class="row ${active && active.v === x.value ? "active" : active ? "dim" : ""}" data-q="${q.question_id}" data-v="${esc(x.value)}" title="${esc(x.label)}">
          <span class="label">${esc(x.label)}</span><span class="bar"><i class="${x.count ? "some" : ""}" style="width:${(100 * x.count / max).toFixed(1)}%"></i></span><span class="n">${fmt(x.count)}</span>
        </div>`).join("");
      const more = all ? "" : `<button class="showmore" data-more="${q.question_id}">Show ${q.rows.length - 5} more</button>`;
      const avg = q.average ? `<span class="avg">Average ${q.average} of ${q.of}</span>` : "";
      return `<div class="cell"><div class="cell-head"><h3>${esc(q.name)}</h3>${avg}</div>${q.answered ? rows + more : `<div class="empty-cell">Not run yet</div>`}</div>`;
    }).join("");
  }

  function answerTags(item) {
    const out = [];
    for (const q of S.project.questions) {
      const a = item.answers[q.id];
      if (!a) continue;
      if (q.type === "noul") { if (a.bucket === "yes") out.push(`<span class="tag yes ${Math.abs(a.noul - q.threshold) < 0.12 ? "unsure" : ""}" title="${(a.noul * 100).toFixed(0)}% sure">${esc(q.name)}</span>`); }
      else if (q.type === "choice") out.push(`<span class="tag ${a.confidence < 0.4 ? "unsure" : ""}" title="${esc(q.name)} · confidence ${(a.confidence * 100).toFixed(0)}%">${esc(a.choice)}</span>`);
      else out.push(`<span class="tag" title="${esc(q.name)}">${esc(q.name)} ${(a.score + 1).toFixed(1)}/${(q.criteria || []).length}</span>`);
    }
    return out.join("");
  }

  function stateParts(state) {
    if (typeof state === "string") return { from: "", preview: state, html: `<div class="kv">${esc(state)}</div>` };
    if (Array.isArray(state)) return { from: "", preview: state.join(" · "), html: `<div class="kv">${state.map((v, i) => `<b>${i + 1}.</b> ${esc(typeof v === "string" ? v : JSON.stringify(v))}`).join("\n")}</div>` };
    const find = (...keys) => { for (const k of Object.keys(state)) if (keys.includes(k.toLowerCase())) return state[k]; return ""; };
    const body = find("body", "text", "content", "message", "comment", "description") || Object.values(state).filter((v) => typeof v === "string").join(" · ");
    return { from: find("from", "author", "sender", "user", "name"), preview: body,
             html: `<div class="kv">${Object.entries(state).map(([k, v]) => `<b>${esc(k)}:</b> ${esc(typeof v === "string" ? v : JSON.stringify(v))}`).join("\n")}</div>` };
  }

  function answerDetail(item) {
    return S.project.questions.map((q) => {
      const a = item.answers[q.id];
      if (!a) return "";
      if (q.type === "noul") return `<div class="ans-q">${esc(q.name)} <span>${a.bucket === "yes" ? "Yes" : "No"} · ${(a.noul * 100).toFixed(0)}% yes</span></div>
        <div class="dist win"><span>Probability of yes</span><span class="bar"><i style="width:${a.noul * 100}%"></i></span><span class="n">${(a.noul * 100).toFixed(0)}%</span></div>`;
      const all = Object.entries(a.probabilities || {});
      const probs = q.type === "score" ? all : all.sort((x, y) => y[1] - x[1]).slice(0, 5);   // levels stay in order
      const top = q.type === "choice" ? a.choice : String(Math.round(a.score));
      const name = (k) => q.type === "score" ? `${+k + 1}. ${(q.criteria || [])[+k] ?? ""}` : k;
      const head = q.type === "choice" ? esc(a.choice) : `${(a.score + 1).toFixed(2)} of ${(q.criteria || []).length}`;
      return `<div class="ans-q">${esc(q.name)} <span>${head} · confidence ${(a.confidence * 100).toFixed(0)}%</span></div>` +
        probs.map(([k, p]) => `<div class="dist ${k === top ? "win" : ""}"><span title="${esc(name(k))}" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(name(k))}</span><span class="bar"><i style="width:${p * 100}%"></i></span><span class="n">${(p * 100).toFixed(0)}%</span></div>`).join("");
    }).join("");
  }

  function renderItems() {
    $("#filters").innerHTML = S.filters.map((f, i) => {
      const q = S.project.questions.find((x) => x.id === f.q);
      const row = S.results.questions.find((x) => x.question_id === f.q)?.rows.find((x) => x.value === f.v);
      return `<span class="chip">${esc(q?.name || "?")}: <b>${esc(row?.label || f.v)}</b><button data-unfilter="${i}" aria-label="Remove filter">×</button></span>`;
    }).join("") || `<span class="muted small">All ${esc(S.project.noun || "items")}</span>`;
    $("#items-count").textContent = `${fmt(S.total)} ${noun(S.total)}`;
    const host = $("#items");
    if (!S.items.length) {
      host.innerHTML = `<li class="items-empty">${S.project.item_count ? "Nothing matches." : `No ${esc(S.project.noun || "items")} yet. Use “Add data” to import some.`}</li>`;
    } else {
      host.innerHTML = S.items.map((it) => {
        const parts = stateParts(it.state), open = S.openItem === it.id;
        const detail = open ? `<div class="item-detail"><div>${parts.html}</div><div class="ans">${answerDetail(it) || '<span class="muted">Not sorted yet.</span>'}</div></div>` : "";
        return `<li class="item" data-item="${it.id}"><div class="item-top"><span class="item-title">${esc(it.title)}</span><span class="item-from">${esc(parts.from)}</span></div>
          ${open || String(parts.preview).trim().slice(0, 140) === it.title ? "" : `<div class="item-preview">${esc(String(parts.preview).slice(0, 220))}</div>`}<div class="tags">${answerTags(it)}</div>${detail}</li>`;
      }).join("");
    }
    $("#more").classList.toggle("hidden", S.items.length >= S.total);
  }

  /* ---------- the "Test one" view ---------- */
  const draft = () => (S.drafts[S.pid] ||= { text: "", asked: null, out: null, hashes: {}, error: "", busy: false });

  function renderView() {
    const test = S.view === "test";
    $(".layout").classList.toggle("testing", test);
    $("#view-test").classList.toggle("hidden", !test);
    $("#view-batch").classList.toggle("hidden", test);
    document.querySelectorAll("#view-toggle button").forEach((b) => { b.classList.toggle("on", b.dataset.view === S.view); b.setAttribute("aria-selected", b.dataset.view === S.view); });
  }

  function stateAsText(state) {
    if (typeof state === "string") return state;
    if (Array.isArray(state)) return state.map((v) => typeof v === "string" ? v : JSON.stringify(v)).join("\n");
    return Object.entries(state).map(([k, v]) => `${k}: ${typeof v === "string" ? v : JSON.stringify(v)}`).join("\n");
  }

  /* Pasted JSON objects and arrays are sent as structure, so {"subject": ..., "body": ...} works; anything else is text. */
  function stateFromText(text) {
    const t = text.trim();
    if (/^[\[{]/.test(t)) { try { const v = JSON.parse(t); if (v && typeof v === "object") return v; } catch {} }
    return t;
  }

  /* The textarea grows with its text, between its CSS min-height and 60% of the window. */
  function autosize() {
    const box = $("#ask-text");
    box.style.height = "auto";
    box.style.height = Math.min(box.scrollHeight + 2, Math.round(innerHeight * 0.6)) + "px";
  }

  const recentKey = () => "recent:" + S.pid;
  const recents = () => store.get(recentKey(), []);

  function renderAsk() {
    const d = draft(), box = $("#ask-text");
    if (box.value !== d.text) box.value = d.text;
    box.placeholder = `Paste or type one ${noun(1)}…`;
    autosize();
    $("#ask-sample").classList.toggle("hidden", !S.items.length);
    $("#ask-recent").classList.toggle("hidden", !recents().length);
    $("#ask-copy").disabled = !S.project.questions.length;
    $("#ask-go").disabled = d.busy;
    if (!d.busy) $("#ask-go").textContent = "Ask";
    $("#answers").setAttribute("aria-busy", String(d.busy));
    $("#ask-error").textContent = d.error; $("#ask-error").classList.toggle("hidden", !d.error);
    renderAnswers();
  }

  const YES_TIP = "The model's probability that the answer is yes. The verdict is yes at or above this question's threshold (the tick on the bar).";
  const CONF_TIP = "How clearly one answer beats the rest: 1 means all the probability is on one answer, 0 means a tie.";
  const UNSURE_TIP = "Unsure: a yes/no within 12 points of its threshold, or a confidence under 0.5. Check these by hand or route them to a person.";
  const UNSURE = `<span class="unsure" title="${esc(UNSURE_TIP)}">unsure</span>`;

  function distRow(label, p, win, tick) {
    return `<div class="dist ${win ? "win" : ""}"><span class="l" title="${esc(label)}">${esc(label)}</span>` +
      `<span class="bar"><i style="width:${(p * 100).toFixed(1)}%"></i>${tick == null ? "" : `<b class="tick" style="left:calc(${(tick * 100).toFixed(1)}% - 1px)" title="your yes threshold: ${Math.round(tick * 100)}%"></b>`}</span>` +
      `<span class="n">${Math.round(p * 100)}%</span></div>`;
  }

  function verdictLine(big, sub, subTip, unsure) {
    return `<div class="arow-verdict"><span class="v">${esc(big)}</span>${sub ? `<span class="sub" title="${esc(subTip)}">${sub}</span>` : ""}${unsure ? UNSURE : ""}</div>`;
  }

  function answerBody(q, a) {
    if (q.type === "noul") {
      const yes = a.noul >= q.threshold;
      return verdictLine(yes ? "Yes" : "No", `${Math.round(a.noul * 100)}% yes`, YES_TIP, Math.abs(a.noul - q.threshold) < 0.12) +
        `<div class="dists">${distRow("Probability of yes", a.noul, true, q.threshold)}</div>`;
    }
    const unsure = a.confidence < 0.5;
    if (q.type === "choice") {
      const desc = (q.criteria || {})[a.choice];
      const rows = Object.entries(a.probabilities || {}).sort((x, y) => y[1] - x[1]);
      const all = S.askMore.has(q.id) || rows.length <= 4;
      return verdictLine(a.choice, `confidence ${a.confidence.toFixed(2)}`, CONF_TIP, unsure) +
        (typeof desc === "string" && desc ? `<div class="arow-desc">${esc(desc)}</div>` : "") +
        `<div class="dists">${(all ? rows : rows.slice(0, 3)).map(([k, p]) => distRow(k, p, k === a.choice)).join("")}
        ${all ? "" : `<button class="showmore" data-askmore="${q.id}">Show all ${rows.length}</button>`}</div>`;
    }
    const levels = q.criteria || [], at = Math.min(Math.max(Math.round(a.score), 0), levels.length - 1);
    const text = (v) => typeof v === "string" ? v : JSON.stringify(v);
    return verdictLine(text(levels[at] ?? ""), `level ${at + 1} of ${levels.length} · expected ${(a.score + 1).toFixed(1)} · confidence ${a.confidence.toFixed(2)}`, CONF_TIP, unsure) +
      `<div class="dists">${levels.map((lv, i) => distRow(`${i + 1}. ${text(lv)}`, (a.probabilities || {})[i] ?? 0, i === at)).join("")}</div>`;
  }

  /* One list, one row per question in the project's order: the question on the left, its answer on the right. */
  function renderAnswers() {
    const d = draft(), host = $("#answers"), list = S.project.questions;
    host.classList.toggle("stale", !!d.out && (d.text.trim() !== (d.asked ?? "").trim() || d.out.model !== S.model));
    if (!list.length) {
      host.innerHTML = `<div class="answers-empty"><h3>No questions yet</h3><p>A question is one narrow judgment about the ${esc(noun(1))}: a yes/no, a category, or a score.</p>
        <button class="btn btn-primary" data-newq>+ New question</button></div>`;
      return;
    }
    /* An answer counts only if it was given to this exact wording; renaming or moving the threshold keeps it. */
    const answered = (q) => (d.out && d.hashes[q.id] === q.spec_hash ? d.out.answers[q.id] : null);
    const readout = d.out ? `${fmt(d.out.elapsed_ms)} ms · ${esc(d.out.model)}` + (d.out.truncated ? " · input shortened to fit" : "")
      : `${list.length} question${list.length === 1 ? "" : "s"}, not asked yet`;
    const explain = S.explain ? `<p class="answers-explain" id="answers-explain">${esc(YES_TIP)} ${esc(CONF_TIP)} ${esc(UNSURE_TIP)}</p>` : "";
    host.innerHTML = `<div class="answers-head"><h2>Answers</h2><span class="muted small">${readout}</span>
        <button class="infobtn" data-explain aria-expanded="${S.explain}" aria-controls="answers-explain" title="What the numbers mean" aria-label="What the numbers mean">?</button></div>${explain}` +
      list.map((q) => {
        const a = answered(q);
        const empty = d.out ? "Ask again to answer this one." : "";
        return `<article class="arow ${a ? "has" : ""}" data-qid="${q.id}"><div class="arow-q"><div class="arow-name"><h3>${esc(q.name)}</h3><span class="typechip">${TYPE_LABEL[q.type]}</span>
          <button class="q-edit" data-editq="${q.id}" title="Edit question" aria-label="Edit ${esc(q.name)}"><svg class="ico" viewBox="0 0 24 24"><path d="M4 20l1-4L16.5 4.5a2.1 2.1 0 013 3L8 19l-4 1z"/></svg></button></div>
          <p class="arow-ins">${esc(q.instructions || "")}</p></div>
          <div class="arow-a">${a ? answerBody(q, a) : `<div class="arow-verdict"><span class="v none">—</span><span class="sub">${empty}</span></div>`}</div></article>`;
      }).join("") + `<button class="arow-new" data-newq>+ New question</button>`;
  }

  async function ask() {
    const d = draft(), pid = S.pid;
    if (d.busy) return;
    d.text = $("#ask-text").value;
    if (!d.text.trim()) { d.error = `Type or paste one ${noun(1)} first.`; renderAsk(); $("#ask-text").focus(); return; }
    d.busy = true; d.error = ""; renderAsk();
    const started = Date.now(), label = () => { if (S.pid === pid) $("#ask-go").textContent = `Asking… ${Math.round((Date.now() - started) / 1000)} s`; };
    label(); const ticker = setInterval(label, 1000);            // the first ask can take a while: a model may be loading
    try {
      const asked = d.text, hashes = Object.fromEntries(S.project.questions.map((q) => [q.id, q.spec_hash]));
      const out = await api(`/api/projects/${pid}/ask`, { method: "POST", body: { state: stateFromText(asked), model: S.model } });
      Object.assign(d, { out, asked, hashes });
      store.set("recent:" + pid, [asked.trim(), ...store.get("recent:" + pid, []).filter((t) => t !== asked.trim())].slice(0, 10));
    } catch (err) { d.error = err.message; }
    clearInterval(ticker);
    d.busy = false;
    if (S.pid === pid) renderAsk();
  }

  /* ---------- recent prompts ---------- */
  function openRecent() {
    const menu = $("#recent-menu"), list = recents();
    menu.innerHTML = (list.length ? list.map((t, i) => `<button role="menuitem" data-recent="${i}" title="${esc(t.slice(0, 400))}">${esc(t.replace(/\s+/g, " ").slice(0, 90))}</button>`).join("")
      + `<button role="menuitem" class="muted-item" data-clear-recent>Clear recent prompts</button>` : `<div class="empty">Nothing asked yet in this project.</div>`);
    menu.hidden = false; $("#ask-recent").setAttribute("aria-expanded", "true");
    menu.querySelector("button")?.focus();
  }
  function closeRecent(refocus) {
    if ($("#recent-menu").hidden) return;
    $("#recent-menu").hidden = true; $("#ask-recent").setAttribute("aria-expanded", "false");
    if (refocus) $("#ask-recent").focus();
  }

  /* ---------- copy as API call: the exact /v1/systemone request the Jev-compatible API takes ---------- */
  /* Mirrors store.wire_question: name and threshold are UI-only; empty noul criteria are dropped. */
  function wireOf(q) {
    const w = { type: q.type, instructions: q.instructions || null };
    if (q.type === "noul") {
      const c = Object.fromEntries(Object.entries(q.criteria || {}).filter(([, v]) => v));
      if (Object.keys(c).length) w.criteria = c;
    } else w.criteria = q.criteria;
    return w;
  }
  function apiRequest() {
    const used = new Set(), questions = {};
    for (const q of S.project.questions) {
      let key = q.name.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "") || "question", n = 2;
      while (used.has(key)) key = key.replace(/_\d+$/, "") + "_" + n++;
      used.add(key); questions[key] = wireOf(q);
    }
    const text = $("#ask-text").value.trim();
    return { state: text ? stateFromText(text) : `Replace this with the ${noun(1)} to judge.`, model: S.model, questions };
  }
  function py(v, ind = "") {
    const n = ind + "    ";
    if (v === null || v === undefined) return "None";
    if (v === true) return "True";
    if (v === false) return "False";
    if (typeof v === "string" || typeof v === "number") return JSON.stringify(v);
    if (Array.isArray(v)) return v.length ? `[\n${v.map((x) => n + py(x, n)).join(",\n")},\n${ind}]` : "[]";
    const e = Object.entries(v);
    return e.length ? `{\n${e.map(([k, x]) => `${n}${JSON.stringify(k)}: ${py(x, n)}`).join(",\n")},\n${ind}}` : "{}";
  }
  function snippets() {
    const body = apiRequest(), url = `${location.origin}/v1/systemone`;
    const curl = `curl -s ${url} \\\n  -H "Authorization: Bearer local" \\\n  -H "Content-Type: application/json" \\\n  -d @- <<'JSON'\n${JSON.stringify(body, null, 2)}\nJSON`;
    const python = `# pip install typesafe-sdk\nfrom typesafe_sdk import TypeSafeClient\n\nclient = TypeSafeClient(base_url=${JSON.stringify(location.origin)}, api_key="local")\n\n` +
      `STATE = ${py(body.state)}\n\nQUESTIONS = ${py(body.questions)}\n\nresult = client.system_one(state=STATE, questions=QUESTIONS, model=${JSON.stringify(body.model)})\n` +
      `print(result.raw_http_response.json()["answers"])`;
    return { curl, python };
  }
  async function copyText(text) {
    try { await navigator.clipboard.writeText(text); return true; } catch { return false; }
  }
  function copyModal() {
    const code = snippets(), n = S.project.questions.length;
    let tab = store.get("copyTab", "curl");
    const m = modal(`<div class="modal-head"><h2>Copy as API call</h2><button class="x" data-close aria-label="Close">×</button></div>
      <div class="modal-body"><p class="muted small" style="margin-bottom:12px">The text on the left and all ${n} question${n === 1 ? "" : "s"} of this project, as one request to the
        Jev-compatible endpoint <code>POST /v1/systemone</code> with <b>${esc(S.model)}</b>. It is the request the official <code>typesafe-sdk</code> sends.
        Any API key works unless the server was started with <code>LOCAL_JEV_API_KEY</code>.</p>
        <div class="viewseg codetabs" role="tablist"><button data-tab="curl" role="tab">curl</button><button data-tab="python" role="tab">Python</button></div>
        <pre class="codebox" id="code-view" tabindex="0"></pre></div>
      <div class="modal-foot"><span id="copy-status" class="muted small" role="status"></span><span class="spacer"></span>
        <button class="btn" data-close>Close</button><button class="btn btn-primary" id="copy-go">Copy</button></div>`);
    const R = m.root, show = () => {
      $("#code-view", R).textContent = code[tab];
      R.querySelectorAll("[data-tab]").forEach((b) => { b.classList.toggle("on", b.dataset.tab === tab); b.setAttribute("aria-selected", b.dataset.tab === tab); });
    };
    R.addEventListener("click", async (e) => {
      const t = e.target.closest("[data-tab]");
      if (t) { tab = t.dataset.tab; store.set("copyTab", tab); show(); $("#copy-status", R).textContent = ""; }
      if (e.target.closest("#copy-go")) {
        const ok = await copyText(code[tab]);
        if (!ok) { const range = document.createRange(); range.selectNodeContents($("#code-view", R)); getSelection().removeAllRanges(); getSelection().addRange(range); }
        $("#copy-status", R).textContent = ok ? "Copied." : "Selected: press ⌘C / Ctrl+C to copy.";
      }
    });
    show(); $("#copy-go", R).focus();
  }

  /* ---------- model picker ----------
     Two buttons (#model in "Sort many", #ask-model in "Test one") share one selection and one panel.
     The panel element is permanent and only its contents are re-rendered, so its listeners, bound once
     at startup, can never outlive what they control. */
  const PICKERS = ["#model", "#ask-model"];
  const P = { open: false, anchor: null, active: -1, list: [], sort: store.get("pickSort", "default"), info: false };
  const pct0 = (x) => Math.round(x * 100) + "%";
  const evalOf = (m) => (m && m.evaluation && typeof m.evaluation.accuracy === "number") ? m.evaluation : null;
  const ms = (x) => x == null ? "" : `${fmt(Math.round(x))} ms`;

  function renderPickers() {
    const m = S.models.find((x) => x.name === S.model), e = evalOf(m);
    const meta = e ? [pct0(e.accuracy), ms(e.p50_ms)].filter(Boolean).join(" · ") : "";
    for (const id of PICKERS) {
      const b = $(id);
      b.innerHTML = `<span class="mpick-name">${esc(S.model || "Choose a model")}</span>${meta ? `<span class="mpick-meta">${esc(meta)}</span>` : ""}`;
      b.title = e ? `${S.model}: ${pct0(e.accuracy)} accurate, median ${ms(e.p50_ms) || "unknown"}` : (S.model || "");
      b.setAttribute("aria-expanded", String(P.open && P.anchor === b));
    }
  }

  function sortedModels() {
    const list = S.models.slice();                        // server order = priority; Array.sort is stable, so ties keep it
    if (P.sort === "accuracy") list.sort((a, b) => (evalOf(b)?.accuracy ?? -1) - (evalOf(a)?.accuracy ?? -1));
    if (P.sort === "speed") list.sort((a, b) => (evalOf(a)?.p50_ms ?? Infinity) - (evalOf(b)?.p50_ms ?? Infinity));
    return list;
  }

  function referenceNote(ref) {
    const [name, acc] = Object.entries(ref || {})[0] || [];
    if (typeof acc !== "number") return "";
    return `${/^jev/i.test(name) ? "Jev (hosted)" : esc(name)} scores ${pct0(acc)} on the same items`;
  }

  function renderPanel() {
    const panel = $("#mpick-panel"), list = P.list = sortedModels();
    const measured = S.models.map(evalOf).filter(Boolean), first = measured[0];
    const ref = measured.map((e) => referenceNote(e.reference)).find(Boolean);
    const scope = first ? `Accuracy on ${esc(first.benchmark || "the benchmark")}${first.items ? ` (${fmt(first.items)} items)` : ""} · median latency on this machine`
                        : "These models have not been benchmarked yet.";
    const sorts = measured.length ? `<div class="mpick-sort">Sort:${["default", "accuracy", "speed"].map((k) =>
      `<button type="button" data-sort="${k}" class="${P.sort === k ? "on" : ""}">${k}</button>`).join("")}</div>` : "";
    const explain = P.info ? `<div class="mpick-explain" id="mpick-explain">Accuracy is the share of the benchmark's public decisions each model answers correctly
      (its highest-probability answer). Latency is the median time per request, one decision each, measured through the local API on the machine named in each row's tooltip.
      Hover a row for the 95th-percentile latency and when it was measured.</div>` : "";
    const rows = list.map((m, i) => {
      const e = evalOf(m), sel = m.name === S.model, dis = !!m.unavailable;
      const tip = dis ? m.unavailable : e ? [[e.benchmark, e.items && `${fmt(e.items)} items`].filter(Boolean).join(", "), e.p95_ms != null && `p95 ${ms(e.p95_ms)}`,
        e.hardware, e.measured && `measured ${e.measured}`].filter(Boolean).join(" · ") : "Not benchmarked yet";
      const metrics = dis ? `<span class="mpick-why">${esc(m.unavailable)}</span>`
        : e ? `<span class="mpick-acc"><span class="mpick-bar"><i style="width:${(e.accuracy * 100).toFixed(1)}%"></i></span><b>${pct0(e.accuracy)}</b></span><span class="mpick-ms">${ms(e.p50_ms) || "–"}</span>`
        : `<span class="mpick-none">not measured</span>`;
      return `<div class="mpick-row ${sel ? "sel" : ""} ${dis ? "dis" : ""} ${i === P.active ? "active" : ""}" role="option" id="mpick-opt-${i}" data-i="${i}"
        aria-selected="${sel}" aria-disabled="${dis}" title="${esc(tip)}"><span class="mpick-check" aria-hidden="true">${sel ? "✓" : ""}</span>
        <span class="mpick-main"><span class="mpick-title"><b>${esc(m.name)}</b><span class="typechip">${esc(m.backend)}</span></span>
        <span class="mpick-desc">${esc(m.description || "")}</span></span><span class="mpick-metrics">${metrics}</span></div>`;
    }).join("");
    panel.innerHTML = `<div class="mpick-head"><div class="top"><span>${scope}</span>
      <button type="button" class="mpick-info" data-info aria-expanded="${P.info}" aria-controls="mpick-explain" title="How these are measured" aria-label="How these are measured">i</button></div>
      ${ref ? `<div class="ref">${ref}</div>` : ""}${explain}${sorts}</div><div class="mpick-list">${rows}</div>
      <div class="mpick-foot"><a href="#models">Compare models →</a></div>`;
    panel.setAttribute("aria-activedescendant", P.active >= 0 ? `mpick-opt-${P.active}` : "");
  }

  function placePanel(keepSide = false) {
    const panel = $("#mpick-panel"), r = P.anchor.getBoundingClientRect(), vw = document.documentElement.clientWidth, vh = innerHeight;
    const width = Math.min(Math.max(r.width, 440), vw - 16);
    panel.style.width = width + "px";
    panel.style.left = Math.min(Math.max(8, r.left), vw - width - 8) + "px";
    panel.style.top = panel.style.bottom = ""; panel.style.maxHeight = "none";
    const want = Math.min(panel.scrollHeight, 460), below = vh - r.bottom - 12, above = r.top - 12;
    if (!keepSide || !P.side) P.side = below >= want || below >= above ? "below" : "above";   // do not jump sides while open
    if (P.side === "below") { panel.style.top = r.bottom + 4 + "px"; panel.style.maxHeight = Math.max(140, below) + "px"; }
    else { panel.style.bottom = vh - r.top + 4 + "px"; panel.style.maxHeight = Math.max(140, above) + "px"; }
  }

  function openPicker(btn) {
    const panel = $("#mpick-panel");
    P.open = true; P.anchor = btn; P.info = false;
    P.active = Math.max(0, sortedModels().findIndex((m) => m.name === S.model));
    renderPanel(); panel.hidden = false; placePanel(); renderPickers();
    panel.focus({ preventScroll: true });
    $(`#mpick-opt-${P.active}`)?.scrollIntoView({ block: "nearest" });
  }

  function closePicker(refocus) {
    if (!P.open) return;
    const btn = P.anchor;
    P.open = false; $("#mpick-panel").hidden = true; renderPickers();
    if (refocus) btn.focus();
  }

  function moveActive(step) {
    const n = P.list.length; if (!n) return;
    let i = P.active;
    for (let k = 0; k < n; k++) { i = (i + step + n) % n; if (!P.list[i].unavailable) break; }
    P.active = i; renderPanel(); $(`#mpick-opt-${i}`)?.scrollIntoView({ block: "nearest" });
  }

  async function chooseModel(name) {
    const m = S.models.find((x) => x.name === name);
    if (!m || m.unavailable) return;
    closePicker(true);
    if (name === S.model) return;
    S.model = name; store.set("model", name); renderPickers();
    if (S.pid) { renderAnswers(); await loadResults(); await loadItems(true); }
  }

  /* ---------- the Models page (#models, #models/<name>) ----------
     Everything on it is derived from GET /api/meta, so it stays right when cards change: the guide picks
     models by their numbers, and cards with priority <= 1 (experiments) are listed but kept out of the
     charts and the guide. Charts are hand-written SVG sized to their container and redrawn on resize. */
  /* Jev (hosted) per difficulty tier on JevBench's public items: a published reference run of jev-1.13.0
     (easy 48/48, standard 71/72, hard 81/111). The overall figure comes from each card's evaluation.reference. */
  const JEV_TIERS = { easy: 1.0, standard: 71 / 72, hard: 81 / 111 };
  const TIERS = [["easy", "Easy"], ["standard", "Standard"], ["hard", "Hard"]];
  const NS = "http://www.w3.org/2000/svg";
  const isExperiment = (m) => (m.priority ?? 10) <= 1;
  const charted = () => S.models.filter((m) => evalOf(m) && !isExperiment(m));
  const jevOverall = () => S.models.map((m) => evalOf(m)?.reference?.["jev-1.13.0"]).find((x) => typeof x === "number");
  const pct1 = (x) => `${(Math.round(x * 1000) / 10).toFixed(1).replace(/\.0$/, "")}%`;       // one decimal where a point matters
  const INTERACTIVE_MS = 250;                                   // "interactive speed": median answer under a quarter second
  const gb = (x) => x == null ? "" : `${x < 10 ? x.toFixed(1).replace(/\.0$/, "") : Math.round(x)} GB`;
  const licenceShort = (l) => (l || "").split(" (")[0];
  const anchorOf = (name) => `#models/${encodeURIComponent(name)}`;

  function showModelsPage(on) {
    S.page = on ? "models" : "app";
    const has = S.projects.length > 0;
    $("#models-page").classList.toggle("hidden", !on);
    $("#main").classList.toggle("hidden", on || !has);
    $(".sidebar").classList.toggle("hidden", on || !has);
    $("#welcome").classList.toggle("hidden", on || has);
    $(".layout").classList.toggle("modelsview", on);
    $("#models-link").classList.toggle("on", on);
    $("#models-link").setAttribute("aria-current", on ? "page" : "false");
    document.querySelectorAll("#tabs .tab").forEach((t) => t.classList.toggle("active", !on && +t.dataset.pid === S.pid));
    if (!on) { $("#viz-tip").hidden = true; renderView(); }
  }

  function leaveModels() {
    if (location.hash.startsWith("#models")) history.pushState(null, "", location.pathname + location.search);
    showModelsPage(false);
  }

  async function route() {
    const match = location.hash.match(/^#models(?:\/(.+))?$/);
    if (!match) { if (S.page === "models") showModelsPage(false); return; }
    const first = S.page !== "models";
    showModelsPage(true);
    if (first) {
      try { const meta = await api("/api/meta"); S.models = meta.models; renderPickers(); } catch (err) { toast(err.message); }
      renderModelsPage();
    }
    const name = match[1] ? decodeURIComponent(match[1]) : null, card = name && document.getElementById("model-" + name);
    if (card) {
      card.scrollIntoView({ block: "start", behavior: first ? "auto" : "smooth" });
      card.classList.add("flash"); setTimeout(() => card.classList.remove("flash"), 1400);
    } else if (first || !name) window.scrollTo(0, 0);
  }

  function guide() {
    const ok = charted().filter((m) => !m.unavailable), pick = [];
    if (!ok.length) return "";
    const by = (f, dir = 1) => ok.slice().sort((a, b) => dir * (f(a) - f(b)))[0];
    const line = (m) => `${pct1(evalOf(m).accuracy)} accurate · ${ms(evalOf(m).p50_ms)} median`;
    const best = by((m) => evalOf(m).accuracy, -1), fastest = by((m) => evalOf(m).p50_ms ?? Infinity);
    pick.push(["Most accurate", best, line(best)]);
    if (fastest !== best) pick.push(["Fastest", fastest, line(fastest)]);
    const close = ok.filter((m) => evalOf(best).accuracy - evalOf(m).accuracy <= 0.05);
    const balance = close.sort((a, b) => (evalOf(a).p50_ms ?? Infinity) - (evalOf(b).p50_ms ?? Infinity))[0];
    if (balance !== best && balance !== fastest) pick.push(["Fastest within 5 points of the best", balance, line(balance)]);
    const quick = ok.filter((m) => (evalOf(m).p50_ms ?? Infinity) <= INTERACTIVE_MS).sort((a, b) => evalOf(b).accuracy - evalOf(a).accuracy)[0];
    if (quick && ![best, fastest, balance].includes(quick)) pick.push([`Most accurate at interactive speed (median under ${INTERACTIVE_MS} ms)`, quick, line(quick)]);
    const sized = S.models.filter((m) => !isExperiment(m) && !m.unavailable && m.size?.download_gb != null);
    const mem = S.models.filter((m) => !isExperiment(m) && !m.unavailable && m.size?.memory_gb != null);
    const small = sized.slice().sort((a, b) => a.size.download_gb - b.size.download_gb)[0];
    const light = mem.slice().sort((a, b) => a.size.memory_gb - b.size.memory_gb)[0];
    if (small && small === light) {
      pick.push(["Smallest, for small disks and machines without a large GPU", small, `${gb(small.size.download_gb)} to download, about ${gb(small.size.memory_gb)} in memory`]);
    } else {
      if (small) pick.push(["Smallest download", small, `${gb(small.size.download_gb)} to download` + (evalOf(small) ? ` · ${pct1(evalOf(small).accuracy)} accurate` : "")]);
      if (light) pick.push(["Lightest on memory, for machines without a large GPU", light, `about ${gb(light.size.memory_gb)} in memory`]);
    }
    const jev = jevOverall();
    const gap = jev != null ? `<p class="mp-note">No local model matches Jev (hosted) yet: it scores ${pct1(jev)} on the same items, ${((jev - evalOf(best).accuracy) * 100).toFixed(1)} points above ${esc(best.name)}. Local models win on privacy, cost and latency; use the confidence each answer carries to send the unsure ones elsewhere.</p>` : "";
    return `<section class="mp-section"><h2>How to choose</h2><div class="mp-guide">${pick.map(([k, m, d]) =>
      `<div class="card mp-pick"><div class="k">${esc(k)}</div><div class="v"><a href="${anchorOf(m.name)}">${esc(m.name)}</a></div><div class="d">${esc(d)}</div></div>`).join("")}</div>${gap}</section>`;
  }

  function el(tag, attrs = {}, text) {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) if (v != null) node.setAttribute(k, v);
    if (text != null) node.textContent = text;
    return node;
  }

  function scatter(host) {
    const models = charted(), jev = jevOverall();
    const W = Math.max(300, Math.round(host.clientWidth)), H = W < 460 ? 260 : 300, m = { l: 44, r: 14, t: 18, b: 40 };
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "viz", role: "img", "aria-labelledby": "sc-title sc-desc" });
    svg.append(el("title", { id: "sc-title" }, "Accuracy against median latency"),
               el("desc", { id: "sc-desc" }, models.map((x) => `${x.name}: ${pct0(evalOf(x).accuracy)}, ${ms(evalOf(x).p50_ms)}`).join("; ") + (jev != null ? `; Jev (hosted) ${pct0(jev)}` : "")));
    host.replaceChildren(svg);
    if (!models.length) return;
    const lat = models.map((x) => evalOf(x).p50_ms).filter((v) => v > 0);
    const lo = Math.log10(Math.min(...lat) / 1.8), hi = Math.log10(Math.max(...lat) * 1.8);
    const X = (v) => m.l + (Math.log10(v) - lo) / (hi - lo) * (W - m.l - m.r);
    const accs = models.map((x) => evalOf(x).accuracy).concat(jev != null ? [jev] : []);
    const y0 = Math.max(0, Math.floor((Math.min(...accs) * 100 - 8) / 10) * 10), y1 = 100;
    const Y = (a) => m.t + (1 - (a * 100 - y0) / (y1 - y0)) * (H - m.t - m.b);
    for (let v = y0; v <= y1; v += 10) {
      svg.append(el("line", { x1: m.l, x2: W - m.r, y1: Y(v / 100), y2: Y(v / 100), class: "grid" }));
      svg.append(el("text", { x: m.l - 6, y: Y(v / 100) + 4, "text-anchor": "end" }, `${v}%`));
    }
    const ticks = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000].filter((t) => Math.log10(t) >= lo && Math.log10(t) <= hi);
    for (const t of ticks) {
      svg.append(el("line", { x1: X(t), x2: X(t), y1: m.t, y2: H - m.b, class: "grid" }));
      svg.append(el("text", { x: X(t), y: H - m.b + 15, "text-anchor": "middle" }, t >= 1000 ? `${t / 1000} s` : `${t} ms`));
    }
    svg.append(el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axis" }));
    svg.append(el("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle" }, "Median latency per request (log scale)"));
    if (jev != null) {
      svg.append(el("line", { x1: m.l, x2: W - m.r, y1: Y(jev), y2: Y(jev), class: "ref" }));
      svg.append(el("text", { x: W - m.r, y: Y(jev) - 6, "text-anchor": "end", class: "lbl" }, `Jev (hosted) ${pct0(jev)}`));
    }
    // labels: try right, left, above and below each dot and keep the first spot that overlaps no other label,
    // no dot and stays inside the plot; if none is free, step further down until one is
    const pts = models.map((x) => ({ x: X(evalOf(x).p50_ms), y: Y(evalOf(x).accuracy), m: x })).sort((a, b) => a.y - b.y);
    const boxes = pts.map((p) => ({ x0: p.x - 7, x1: p.x + 7, y0: p.y - 7, y1: p.y + 7 }));
    if (jev != null) boxes.push({ x0: W - m.r - 110, x1: W - m.r, y0: Y(jev) - 18, y1: Y(jev) - 2 });
    const hits = (b) => b.x0 < m.l || b.x1 > W - m.r || b.y0 < m.t - 4 || b.y1 > H - m.b ||
      boxes.some((o) => b.x0 < o.x1 && b.x1 > o.x0 && b.y0 < o.y1 && b.y1 > o.y0);
    for (const p of pts) {
      const w = p.m.name.length * 6.4 + 4, h = 14;
      const spots = [[p.x + 10, p.y - h / 2, "start"], [p.x - 10 - w, p.y - h / 2, "end"], [p.x - w / 2, p.y - 10 - h, "middle"], [p.x - w / 2, p.y + 10, "middle"]];
      for (let dy = 16; dy < 90; dy += 14) spots.push([p.x + 10, p.y - h / 2 + dy, "start"], [p.x - 10 - w, p.y - h / 2 + dy, "end"]);
      const [bx, by, anchor] = spots.find(([sx, sy]) => !hits({ x0: sx, x1: sx + w, y0: sy, y1: sy + h })) || spots[0];
      boxes.push({ x0: bx, x1: bx + w, y0: by, y1: by + h });
      p.lx = anchor === "start" ? bx : anchor === "end" ? bx + w : bx + w / 2; p.ly = by + 11; p.anchor = anchor;
      if (Math.abs(p.ly - 4 - p.y) > 12 || anchor === "middle") p.leader = true;
    }
    for (const p of pts) {
      const e = evalOf(p.m), tip = `<b>${esc(p.m.name)}</b><br>${pct0(e.accuracy)} accurate on ${esc(e.benchmark || "the benchmark")}<br>median ${ms(e.p50_ms)} · p95 ${ms(e.p95_ms) || "–"}`;
      const g = el("g");
      g.append(el("circle", { cx: p.x, cy: p.y, r: 13, class: "hit", tabindex: 0, "data-vtip": tip, "aria-label": `${p.m.name}: ${pct0(e.accuracy)}, ${ms(e.p50_ms)}` }),
               el("circle", { cx: p.x, cy: p.y, r: 6, class: "dot" }),
               el("text", { x: p.lx, y: p.ly, "text-anchor": p.anchor, class: "lbl" }, p.m.name));
      if (p.leader && p.anchor !== "middle") g.prepend(el("line", { x1: p.x, y1: p.y, x2: p.anchor === "start" ? p.lx - 2 : p.lx + 2, y2: p.ly - 4, class: "grid" }));
      svg.append(g);
    }
  }

  function tiers(host) {
    const models = charted().filter((x) => evalOf(x).by_tier).sort((a, b) => evalOf(b).accuracy - evalOf(a).accuracy);
    const W = Math.max(300, Math.round(host.clientWidth)), labelW = Math.min(150, Math.round(W * 0.3)), gap = 12;
    const panelW = (W - labelW - gap * 3) / 3, rowH = 24, top = 36, H = top + models.length * rowH + 6;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "viz", role: "img", "aria-labelledby": "ti-title ti-desc" });
    svg.append(el("title", { id: "ti-title" }, "Accuracy by difficulty tier"),
               el("desc", { id: "ti-desc" }, models.map((x) => `${x.name}: ` + TIERS.map(([k, l]) => `${l} ${pct0(evalOf(x).by_tier[k] ?? 0)}`).join(", ")).join("; ")));
    host.replaceChildren(svg);
    TIERS.forEach(([key, label], i) => {
      const x0 = labelW + gap + i * (panelW + gap), X = (a) => x0 + a * panelW;
      svg.append(el("text", { x: x0, y: 13, class: "ptitle" }, label));
      svg.append(el("line", { x1: x0, x2: x0, y1: top - 8, y2: H - 4, class: "axis" }));
      const jev = JEV_TIERS[key];
      const showJev = jev != null && jevOverall() != null;
      if (showJev) svg.append(el("text", { x: X(jev) + 2, y: 28, "text-anchor": "end" }, `Jev ${pct0(jev)}`));
      models.forEach((mm, r) => {
        const a = evalOf(mm).by_tier[key];
        if (a == null) return;
        const cy = top + r * rowH + rowH / 2, w = Math.max(1, a * panelW);
        const clash = showJev && x0 + w + 4 < X(jev) + 3 && x0 + w + 32 > X(jev) - 3;                // an outside label would hit the Jev tick
        const inside = w > panelW - 34 || (clash && w > 34);
        svg.append(el("rect", { x: x0, y: cy - 5, width: w, height: 10, rx: 3, class: "bar" }));
        if (showJev) svg.append(el("line", { x1: X(jev), x2: X(jev), y1: cy - 8, y2: cy + 8, class: "ref" }));
        let lx = inside ? x0 + w - 4 : x0 + w + 4;
        if (!inside && clash) lx = X(jev) + 5;                                                         // short bar: step past the tick
        svg.append(el("text", { x: lx, y: cy + 4, "text-anchor": inside ? "end" : "start", class: "val", style: inside ? "fill: var(--panel)" : null }, pct0(a)));
      });
    });
    models.forEach((mm, r) => {
      const cy = top + r * rowH + rowH / 2, e = evalOf(mm);
      const tip = `<b>${esc(mm.name)}</b> · ${pct0(e.accuracy)} overall<br>` + TIERS.map(([k, l]) => `${l}: ${pct0(e.by_tier[k] ?? 0)}` + (JEV_TIERS[k] != null ? ` (Jev ${pct0(JEV_TIERS[k])})` : "")).join("<br>");
      svg.insertBefore(el("rect", { x: 0, y: cy - rowH / 2, width: W, height: rowH, rx: 4, class: "rowhit", tabindex: 0, "data-vtip": tip,
                                    "aria-label": `${mm.name}: ` + TIERS.map(([k, l]) => `${l} ${pct0(e.by_tier[k] ?? 0)}`).join(", ") }), svg.children[2]);
      const name = mm.name.length > 22 && labelW < 150 ? mm.name.slice(0, 20) + "…" : mm.name;
      svg.append(el("text", { x: 0, y: cy + 4, class: "lbl" }, name));
    });
  }

  function numbersTable() {
    const rows = charted().sort((a, b) => evalOf(b).accuracy - evalOf(a).accuracy);
    const jev = jevOverall();
    const body = rows.map((m) => { const e = evalOf(m), t = e.by_tier || {};
      return `<tr><td>${esc(m.name)}</td><td>${pct1(e.accuracy)}</td>${TIERS.map(([k]) => `<td>${t[k] == null ? "–" : pct0(t[k])}</td>`).join("")}
        <td>${ms(e.p50_ms) || "–"}</td><td>${ms(e.p95_ms) || "–"}</td><td>${esc(m.size?.params ?? "–")}</td><td>${m.size?.download_gb != null ? gb(m.size.download_gb) : "–"}</td><td>${e.failed ?? "–"}</td></tr>`; }).join("");
    const ref = jev != null ? `<tr><td>Jev (hosted), reference</td><td>${pct1(jev)}</td>${TIERS.map(([k]) => `<td>${pct0(JEV_TIERS[k])}</td>`).join("")}<td>–</td><td>–</td><td>–</td><td>–</td><td>–</td></tr>` : "";
    return `<details class="mp-table"><summary>Show the numbers as a table</summary><div class="wrap"><table>
      <thead><tr><th>Model</th><th>Accuracy</th><th>Easy</th><th>Standard</th><th>Hard</th><th>Median</th><th>p95</th><th>Params</th><th>Download</th><th>Failed</th></tr></thead>
      <tbody>${body}${ref}</tbody></table></div></details>`;
  }

  function modelCard(m) {
    const e = evalOf(m), t = e?.by_tier || {}, sel = m.name === S.model;
    const stat = (k, v, extra = "") => `<div class="mstat"><div class="k">${k}</div><div class="v">${v}</div>${extra}</div>`;
    const stats = e ? [
      stat("Accuracy", `${pct1(e.accuracy)} <small>of ${fmt(e.items || 0)}</small>`, `<div class="bar"><i style="width:${(e.accuracy * 100).toFixed(1)}%"></i></div>`),
      stat("Median latency", ms(e.p50_ms) || "–"), stat("p95 latency", ms(e.p95_ms) || "–"),
      `<div class="mstat"><div class="k">By difficulty</div><div class="mtiers">${TIERS.map(([k, l]) => t[k] == null ? "" :
        `<div class="mtier"><span>${l}</span><span class="t"><i style="width:${(t[k] * 100).toFixed(1)}%"></i></span><span class="n">${pct0(t[k])}</span></div>`).join("")}</div></div>`] : [];
    stats.push(stat("Parameters", esc(m.size?.params ?? "–")), stat("Download", m.size?.download_gb != null ? gb(m.size.download_gb) : "–"),
               stat("Licence", `<small title="${esc(m.license || "")}">${esc(licenceShort(m.license) || "–")}</small>`));
    const list = (items) => items?.length ? `<ul>${items.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : `<p class="none">No guidance yet.</p>`;
    const members = m.members?.length ? `<span>Pools ${m.members.map((n) => `<a href="${anchorOf(n)}">${esc(n)}</a>`).join(" + ")}</span>` : "";
    const measured = e ? `<span>Measured ${esc(e.measured || "")}${e.hardware ? ` on ${esc(e.hardware)}` : ""}${e.failed ? ` · ${e.failed} failed` : ""}</span>` : "";
    const button = m.unavailable ? `<button class="btn" disabled title="${esc(m.unavailable)}">Unavailable</button>`
      : sel ? `<button class="btn" disabled>✓ In use</button>` : `<button class="btn btn-primary" data-use="${esc(m.name)}">Use this model</button>`;
    return `<article class="card mcard" id="model-${esc(m.name)}" aria-labelledby="mt-${esc(m.name)}">
      <div class="mcard-head"><h3 id="mt-${esc(m.name)}">${esc(m.name)}</h3><span class="typechip">${esc(m.backend)}</span>
        ${m.default ? `<span class="badge">server default</span>` : ""}${isExperiment(m) ? `<span class="typechip">experiment</span>` : ""}</div>
      ${m.summary ? `<p class="mcard-sum">${esc(m.summary)}</p>` : ""}<p class="mcard-desc">${esc(m.description || "")}</p>
      ${e ? "" : `<div class="notbench">Not benchmarked yet, so it is left out of the charts above.</div>`}
      <div class="mstats">${stats.join("")}</div>
      <div class="mcard-lists"><div><h4>Use it when</h4>${list(m.use_when)}</div><div><h4>Avoid it when</h4>${list(m.avoid_when)}</div></div>
      <div class="mcard-foot">${members}${measured}${m.unavailable ? `<span>${esc(m.unavailable)}</span>` : ""}<span class="spacer"></span>${button}</div></article>`;
  }

  function renderModelsPage() {
    const page = $("#models-page"), shown = charted(), e0 = evalOf(shown[0]);
    const unbench = S.models.filter((m) => !evalOf(m) && !isExperiment(m)).length, experiments = S.models.filter(isExperiment).length;
    const left = [unbench && `${unbench} not benchmarked yet`, experiments && `${experiments} experimental (priority 1)`].filter(Boolean).join(" and ");
    const hw = e0?.hardware ? ` on an ${esc(e0.hardware)}` : "";
    page.innerHTML = `<a class="mp-back" href="" data-back>← Back to projects</a>
      <h1 id="models-title">Models</h1>
      <p class="mp-intro">Every model answers the same three kinds of question through the same API; they differ in accuracy, speed and size.
        <b>Accuracy</b> is the share of JevBench's ${fmt(e0?.items || 231)} public decisions a model answers correctly (its most probable answer).
        <b>Latency</b> is the median time per request, one decision each, through the local API${hw}.
        JevBench is an independent benchmark, built by Benchmark Heaven, not by this project.</p>
      ${guide()}
      <section class="mp-section"><div class="mp-charts">
        <div class="card mp-chart"><h2>Accuracy against speed</h2><div id="chart-scatter"></div>
          <p class="cap">Up and to the left is better: more accurate and faster. Latency is on a log scale. Hover or focus a dot for its p95.</p></div>
        <div class="card mp-chart"><h2>Accuracy by difficulty</h2><div id="chart-tiers"></div>
          <p class="cap">JevBench's public items in three tiers; the dashed tick on each row marks Jev (hosted). Models are sorted by overall accuracy.</p></div>
      </div>${left ? `<p class="mp-note">Left out of the charts: ${left}.</p>` : ""}${shown.length ? numbersTable() : ""}</section>
      <section class="mp-section"><h2>All models</h2><div class="mp-cards">${S.models.slice().sort((a, b) => isExperiment(a) - isExperiment(b) || (b.priority ?? 10) - (a.priority ?? 10)).map(modelCard).join("")}</div></section>`;
    drawCharts();
  }

  function drawCharts() {
    if (S.page !== "models") return;
    const sc = $("#chart-scatter"), ti = $("#chart-tiers");
    if (!charted().length) { sc.innerHTML = ti.innerHTML = `<p class="mp-note">No benchmarked models yet.</p>`; return; }
    scatter(sc); tiers(ti);
  }

  function showTip(target, x, y) {
    const tip = $("#viz-tip"); tip.innerHTML = target.getAttribute("data-vtip"); tip.hidden = false;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = Math.min(Math.max(8, x + 14), innerWidth - w - 8) + "px";
    tip.style.top = (y + 16 + h > innerHeight ? y - h - 12 : y + 16) + "px";
  }

  /* ---------- modals ---------- */
  function modal(html, { narrow = false } = {}) {
    const root = $("#modal-root");
    root.innerHTML = `<div class="overlay"><div class="modal ${narrow ? "narrow" : ""}" role="dialog" aria-modal="true">${html}</div></div>`;
    const close = () => { root.innerHTML = ""; document.removeEventListener("keydown", onKey); };
    const onKey = (e) => { if (e.key === "Escape") close(); };
    document.addEventListener("keydown", onKey);
    $(".overlay", root).addEventListener("mousedown", (e) => { if (e.target.classList.contains("overlay")) close(); });
    root.querySelectorAll("[data-close]").forEach((b) => b.addEventListener("click", close));
    // Hand back the dialog element, not #modal-root: the root outlives every modal, so listeners
    // bound to it would pile up and keep firing with a previous modal's state.
    return { root: $(".modal", root), close };
  }

  function questionModal(existing) {
    const q = existing ? JSON.parse(JSON.stringify(existing)) : { name: "", type: "noul", instructions: "", criteria: null, threshold: 0.5, enabled: true };
    const draft = { noul: q.type === "noul" ? (q.criteria || {}) : {},
                    choice: q.type === "choice" ? Object.entries(q.criteria || {}).map(([k, d]) => [k, d || ""]) : [["", ""], ["", ""]],
                    score: q.type === "score" ? (q.criteria || []).slice() : ["", "", ""] };
    const one = noun(1);
    const m = modal(`
      <div class="modal-head"><h2>${existing ? "Edit question" : "New question"}</h2><button class="x" data-close aria-label="Close">×</button></div>
      <div class="modal-body">
        <div class="field"><div class="label">Answer type</div><div class="seg" id="q-type">
          <button data-t="noul"><b>Yes / No</b><small>Is it or isn’t it, with how sure.</small></button>
          <button data-t="choice"><b>Category</b><small>Picks one from your list.</small></button>
          <button data-t="score"><b>Score</b><small>Rates it on a scale you define.</small></button></div></div>
        <div class="field"><label for="q-name">Name</label><input id="q-name" class="input" maxlength="120" value="${esc(q.name)}" placeholder="e.g. Invoice or receipt"></div>
        <div class="field"><label for="q-text">Question</label><textarea id="q-text" placeholder="e.g. Is this ${esc(one)} a receipt, invoice, payment confirmation, or billing notice?">${esc(q.instructions)}</textarea>
          <div class="help">Write it out in full. The model reads this and the ${esc(one)}, nothing else.</div></div>
        <div id="q-extra"></div>
        <div id="q-error" class="error"></div>
      </div>
      <div id="q-try" class="tryout hidden"></div>
      <div class="modal-foot">
        ${existing ? `<button class="btn btn-danger" id="q-delete">Delete</button>` : ""}
        <button class="btn" id="q-tryit">Try it on 6 ${esc(noun(6))}</button><span class="spacer"></span>
        <button class="btn" data-close>Cancel</button><button class="btn btn-primary" id="q-save">Save</button>
      </div>`);
    const R = m.root;

    function renderExtra() {
      R.querySelectorAll("#q-type button").forEach((b) => b.classList.toggle("on", b.dataset.t === q.type));
      const host = $("#q-extra", R);
      if (q.type === "noul") {
        host.innerHTML = `<div class="two">
          <div class="field"><label>What counts as yes <em>optional</em></label><textarea id="c-yes">${esc(draft.noul.true || "")}</textarea></div>
          <div class="field"><label>What counts as no <em>optional</em></label><textarea id="c-no">${esc(draft.noul.false || "")}</textarea></div></div>
          <div class="field"><label for="q-th">Call it a yes when the model is at least this sure</label>
            <div class="pct"><input id="q-th" class="input" type="number" min="1" max="99" value="${Math.round(q.threshold * 100)}"><span class="muted">%</span></div>
            <div class="help">You can change this later without running again.</div></div>`;
      } else if (q.type === "choice") {
        host.innerHTML = `<div class="field"><div class="label">Categories <em>name, then when it applies (optional)</em></div><div id="opts">` +
          draft.choice.map(([k, d], i) => `<div class="opt"><span class="num">${i + 1}</span><input class="input" data-k="${i}" value="${esc(k)}" placeholder="Name"><input class="input" data-d="${i}" value="${esc(d)}" placeholder="When does this apply?"><button class="rm" data-rm="${i}" title="Remove">×</button></div>`).join("") +
          `</div><button class="btn btn-sm" id="opt-add">+ Add category</button><div class="help">Add an “Other” category if your list might not cover everything. Up to 255.</div></div>`;
      } else {
        host.innerHTML = `<div class="field"><div class="label">Levels <em>lowest first — describe situations, not degrees</em></div><div id="opts">` +
          draft.score.map((d, i) => `<div class="opt level"><span class="num">${i + 1}</span><input class="input" data-l="${i}" value="${esc(d)}" placeholder="${i === 0 ? "e.g. No action needed at all" : "Describe this level"}"><button class="rm" data-rm="${i}" title="Remove">×</button></div>`).join("") +
          `</div><button class="btn btn-sm" id="opt-add" ${draft.score.length >= 10 ? "disabled" : ""}>+ Add level</button><div class="help">Between 2 and 10 levels. The score is a weighted average, so it can land between two levels.</div></div>`;
      }
    }

    function capture() {
      if (q.type === "noul") { draft.noul = { true: $("#c-yes", R).value.trim(), false: $("#c-no", R).value.trim() }; q.threshold = Math.min(0.99, Math.max(0.01, (+$("#q-th", R).value || 50) / 100)); }
      if (q.type === "choice") draft.choice = draft.choice.map((_, i) => [$(`[data-k="${i}"]`, R).value, $(`[data-d="${i}"]`, R).value]);
      if (q.type === "score") draft.score = draft.score.map((_, i) => $(`[data-l="${i}"]`, R).value);
    }

    function payload() {
      capture();
      const name = $("#q-name", R).value.trim(), instructions = $("#q-text", R).value.trim();
      if (!name) throw new Error("Give the question a name.");
      let criteria = null;
      if (q.type === "noul") { if (!instructions) throw new Error("Write the question out in full."); criteria = draft.noul; }
      if (q.type === "choice") {
        const rows = draft.choice.map(([k, d]) => [k.trim(), d.trim()]).filter(([k]) => k);
        if (rows.length < 2) throw new Error("Add at least two categories.");
        if (new Set(rows.map(([k]) => k)).size !== rows.length) throw new Error("Category names must be unique.");
        criteria = Object.fromEntries(rows.map(([k, d]) => [k, d || null]));
      }
      if (q.type === "score") {
        const levels = draft.score.map((d) => d.trim()).filter(Boolean);
        if (levels.length < 2) throw new Error("Describe at least two levels.");
        criteria = levels;
      }
      return { name, type: q.type, instructions, criteria, threshold: q.threshold, enabled: q.enabled };
    }

    const fail = (err) => { $("#q-error", R).textContent = err.message; };
    R.addEventListener("click", async (e) => {
      const t = e.target.closest("button"); if (!t) return;
      if (t.dataset.t) { capture(); q.type = t.dataset.t; renderExtra(); }
      else if (t.id === "opt-add") { capture(); (q.type === "choice" ? draft.choice.push(["", ""]) : draft.score.push("")); renderExtra(); }
      else if (t.dataset.rm !== undefined) { capture(); (q.type === "choice" ? draft.choice : draft.score).splice(+t.dataset.rm, 1); renderExtra(); }
      else if (t.id === "q-save") {
        try { const body = payload(); $("#q-error", R).textContent = "";
          await api(existing ? `/api/questions/${existing.id}` : `/api/projects/${S.pid}/questions`, { method: existing ? "PUT" : "POST", body });
          m.close(); await refresh(); } catch (err) { fail(err); }
      } else if (t.id === "q-delete") {
        if (!confirm(`Delete “${existing.name}” and its results?`)) return;
        await api(`/api/questions/${existing.id}`, { method: "DELETE" });
        S.filters = S.filters.filter((f) => f.q !== existing.id); m.close(); await refresh();
      } else if (t.id === "q-tryit") {
        const box = $("#q-try", R);
        try { const body = payload(); $("#q-error", R).textContent = "";
          t.disabled = true; box.classList.remove("hidden"); box.innerHTML = `<div class="tryout-row"><span class="muted">Thinking…</span></div>`;
          const out = await api(`/api/projects/${S.pid}/try`, { method: "POST", body: { question: body, model: S.model, n: 6 } });
          box.innerHTML = out.results.map((r) => {
            const a = r.answer;
            const text = body.type === "noul" ? `<b>${a.bucket === "yes" ? "Yes" : "No"}</b> · ${(a.noul * 100).toFixed(0)}% yes`
              : body.type === "choice" ? `<b>${esc(a.choice)}</b> · ${(a.confidence * 100).toFixed(0)}% confident`
              : `<b>${(a.score + 1).toFixed(1)}</b> of ${body.criteria.length} · ${(a.confidence * 100).toFixed(0)}% confident`;
            return `<div class="tryout-row"><span class="t">${esc(r.title)}</span><span class="a">${text}</span></div>`;
          }).join("") + `<div class="tryout-row"><span class="muted small">${out.results.length} ${esc(noun(out.results.length))} in ${out.elapsed}s · ${esc(out.model)} · nothing saved</span></div>`;
        } catch (err) { box.classList.add("hidden"); fail(err); } finally { t.disabled = false; }
      }
    });
    renderExtra();
    $("#q-name", R).focus();
  }

  function projectModal(existing) {
    const p = existing || { name: "", description: "", noun: "items", settings: {} };
    const m = modal(`
      <div class="modal-head"><h2>${existing ? "Project settings" : "New project"}</h2><button class="x" data-close aria-label="Close">×</button></div>
      <div class="modal-body">
        <div class="field"><label for="p-name">Name</label><input id="p-name" class="input" maxlength="80" value="${esc(p.name)}" placeholder="e.g. YouTube comments"></div>
        <div class="field"><label for="p-desc">Description <em>optional</em></label><input id="p-desc" class="input" value="${esc(p.description)}" placeholder="What is in this pile?"></div>
        <div class="two">
          <div class="field"><label for="p-noun">What is one item called? <em>plural</em></label><input id="p-noun" class="input" value="${esc(p.noun)}" placeholder="emails, comments, tickets…"></div>
          <div class="field"><label for="p-tokens">Read up to this many tokens per item</label><input id="p-tokens" class="input" type="number" min="64" max="3500" step="64" value="${p.settings?.max_state_tokens || 1024}">
            <div class="help">Longer items keep their start and end. Lower is faster.</div></div>
        </div>
        <div id="p-error" class="error"></div>
      </div>
      <div class="modal-foot">${existing ? `<button class="btn btn-danger" id="p-delete">Delete project</button><button class="btn btn-danger" id="p-clear">Remove all ${esc(p.noun)}</button>` : ""}
        <span class="spacer"></span><button class="btn" data-close>Cancel</button><button class="btn btn-primary" id="p-save">${existing ? "Save" : "Create"}</button></div>`, { narrow: !existing });
    const R = m.root;
    $("#p-name", R).focus();
    $("#p-save", R).addEventListener("click", async () => {
      try {
        const body = { name: $("#p-name", R).value.trim(), description: $("#p-desc", R).value.trim(), noun: $("#p-noun", R).value.trim().toLowerCase() || "items",
                       settings: { ...(p.settings || {}), max_state_tokens: +$("#p-tokens", R).value || 1024 } };
        if (!body.name) throw new Error("Give the project a name.");
        const saved = await api(existing ? `/api/projects/${existing.id}` : "/api/projects", { method: existing ? "PATCH" : "POST", body });
        m.close(); await loadProjects(saved.id);
        if (!existing) dataModal();
      } catch (err) { $("#p-error", R).textContent = err.message; }
    });
    $("#p-delete", R)?.addEventListener("click", async () => {
      if (!confirm(`Delete the project “${p.name}” with all its questions and results?`)) return;
      await api(`/api/projects/${existing.id}`, { method: "DELETE" }); m.close(); await loadProjects(null);
    });
    $("#p-clear", R)?.addEventListener("click", async () => {
      if (!confirm(`Remove all ${p.noun} from “${p.name}”? Questions are kept.`)) return;
      await api(`/api/projects/${existing.id}/items`, { method: "DELETE" }); m.close(); await refresh();
    });
  }

  function dataModal() {
    const m = modal(`
      <div class="modal-head"><h2>Add ${esc(S.project.noun || "data")}</h2><button class="x" data-close aria-label="Close">×</button></div>
      <div class="modal-body">
        <div class="field"><div id="drop" class="drop"><b>Drop files here</b> or click to choose<div class="small" style="margin-top:4px">CSV · TSV · JSON · JSONL · TXT · .eml · .mbox — nothing leaves this machine</div></div>
          <input id="file" type="file" multiple class="hidden" accept=".csv,.tsv,.json,.jsonl,.ndjson,.txt,.md,.eml,.mbox,text/*"></div>
        <div class="field"><label for="paste">…or paste text</label><textarea id="paste" style="min-height:120px" placeholder="One ${esc(noun(1))} per line, or separate longer ones with a blank line."></textarea></div>
        <div class="field"><label for="split">Split pasted / plain text</label>
          <select id="split" class="select" style="max-width:none;width:100%"><option value="auto">Automatically</option><option value="lines">One per line</option><option value="paragraphs">One per paragraph (blank line between)</option><option value="whole">Keep as a single item</option></select></div>
        <div id="d-error" class="error"></div>
      </div>
      <div class="modal-foot"><span id="d-status" class="muted small"></span><span class="spacer"></span><button class="btn" data-close>Cancel</button><button class="btn btn-primary" id="d-add">Add</button></div>`);
    const R = m.root, drop = $("#drop", R), input = $("#file", R);
    let files = [];
    const show = () => { $("#d-status", R).textContent = files.length ? files.map((f) => f.name).join(", ") : ""; };
    drop.addEventListener("click", () => input.click());
    input.addEventListener("change", () => { files = [...input.files]; show(); });
    drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("over"));
    drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); files = [...e.dataTransfer.files]; show(); });
    $("#d-add", R).addEventListener("click", async (e) => {
      const split = $("#split", R).value, pasted = $("#paste", R).value.trim();
      try {
        if (!files.length && !pasted) throw new Error("Choose a file or paste some text.");
        e.target.disabled = true; let added = 0;
        for (const f of files) added += (await api(`/api/projects/${S.pid}/items/import`, { method: "POST", body: { filename: f.name, content: await f.text(), split } })).added;
        if (pasted) added += (await api(`/api/projects/${S.pid}/items/import`, { method: "POST", body: { filename: "pasted.txt", content: pasted, split } })).added;
        m.close(); toast(`Added ${fmt(added)} ${noun(added)}.`);
        S.projects = await api("/api/projects"); await refresh();
      } catch (err) { $("#d-error", R).textContent = err.message; e.target.disabled = false; }
    });
  }

  /* ---------- events ---------- */
  $("#tabs").addEventListener("click", (e) => { const b = e.target.closest("[data-pid]"); if (b) openProject(+b.dataset.pid); });
  $("#new-project").addEventListener("click", () => projectModal(null));
  $("#project-settings").addEventListener("click", () => projectModal(S.project));
  $("#new-question").addEventListener("click", () => questionModal(null));
  $("#add-data").addEventListener("click", dataModal);

  window.addEventListener("hashchange", route);
  $("#models-page").addEventListener("click", async (e) => {
    const use = e.target.closest("[data-use]");
    if (e.target.closest("[data-back]")) { e.preventDefault(); leaveModels(); return; }
    if (!use) return;
    await chooseModel(use.dataset.use);
    if (S.cameFromApp) history.back(); else leaveModels();          // back to wherever the user came from
  });
  document.addEventListener("mouseover", (e) => { const t = e.target.closest?.("[data-vtip]"); if (t) showTip(t, e.clientX, e.clientY); });
  document.addEventListener("mousemove", (e) => { const t = e.target.closest?.("[data-vtip]"); if (t) showTip(t, e.clientX, e.clientY); });
  document.addEventListener("mouseout", (e) => { if (e.target.closest?.("[data-vtip]") && !e.relatedTarget?.closest?.("[data-vtip]")) $("#viz-tip").hidden = true; });
  document.addEventListener("focusin", (e) => { const t = e.target.closest?.("[data-vtip]");
    if (t) { const r = t.getBoundingClientRect(); showTip(t, r.left + r.width / 2, r.bottom - 6); } else $("#viz-tip").hidden = true; });
  let chartTimer;
  window.addEventListener("resize", () => { clearTimeout(chartTimer); chartTimer = setTimeout(drawCharts, 120); });

  $("#view-toggle").addEventListener("click", (e) => {
    const b = e.target.closest("[data-view]"); if (!b) return;
    S.view = b.dataset.view; store.set("view:" + S.pid, S.view); renderView(); renderAsk();
  });
  $("#ask-go").addEventListener("click", ask);
  $("#ask-text").addEventListener("input", (e) => { draft().text = e.target.value; autosize(); renderAnswers(); });
  $("#ask-copy").addEventListener("click", () => { if (S.project?.questions.length) copyModal(); });
  $("#ask-recent").addEventListener("click", () => ($("#recent-menu").hidden ? openRecent() : closeRecent(true)));
  $("#recent-menu").addEventListener("click", (e) => {
    const item = e.target.closest("[data-recent]");
    if (e.target.closest("[data-clear-recent]")) { store.set(recentKey(), []); closeRecent(false); renderAsk(); $("#ask-text").focus(); return; }
    if (!item) return;
    const d = draft(); d.text = recents()[+item.dataset.recent] || ""; d.error = "";
    closeRecent(false); renderAsk(); $("#ask-text").focus();
  });
  $("#recent-menu").addEventListener("keydown", (e) => {
    const items = [...$("#recent-menu").querySelectorAll("button")], i = items.indexOf(document.activeElement);
    if (e.key === "Escape" || e.key === "Tab") { e.preventDefault(); closeRecent(true); }
    else if (e.key === "ArrowDown") { e.preventDefault(); items[(i + 1) % items.length]?.focus(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); items[(i - 1 + items.length) % items.length]?.focus(); }
  });
  document.addEventListener("mousedown", (e) => { if (!e.target.closest(".menuwrap")) closeRecent(false); });
  window.addEventListener("resize", () => { if (S.view === "test") autosize(); });
  $("#ask-text").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); ask(); } });
  $("#ask-sample").addEventListener("click", () => {
    if (!S.items.length) return;
    const d = draft(), others = S.items.filter((it) => stateAsText(it.state) !== d.text);
    const pick = (others.length ? others : S.items)[Math.floor(Math.random() * (others.length || S.items.length))];
    d.text = stateAsText(pick.state); d.error = ""; renderAsk(); $("#ask-text").focus();
  });
  /* #answers is filled by renderAnswers and never replaced itself, so one delegated listener bound here is safe. */
  $("#answers").addEventListener("click", (e) => {
    const more = e.target.closest("[data-askmore]"), edit = e.target.closest("[data-editq]");
    if (e.target.closest("[data-explain]")) { S.explain = !S.explain; renderAnswers(); $("#answers [data-explain]")?.focus(); }
    else if (more) { S.askMore.add(+more.dataset.askmore); renderAnswers(); }
    else if (edit) questionModal(S.project.questions.find((q) => q.id === +edit.dataset.editq));
    else if (e.target.closest("[data-newq]")) questionModal(null);
  });

  $("#questions").addEventListener("click", async (e) => {
    const li = e.target.closest("li"); if (!li) return;
    const q = S.project.questions.find((x) => x.id === +li.dataset.qid);
    if (e.target.matches("input[type=checkbox]")) {
      q.enabled = e.target.checked;
      await api(`/api/questions/${q.id}/enabled?enabled=${q.enabled}`, { method: "POST" });
      await loadResults();
    } else questionModal(q);
  });

  $("#results").addEventListener("click", async (e) => {
    const more = e.target.closest("[data-more]");
    if (more) { S.showAll.add(+more.dataset.more); renderResults(); return; }
    const row = e.target.closest(".row"); if (!row) return;
    const q = +row.dataset.q, v = row.dataset.v, at = S.filters.findIndex((f) => f.q === q);
    if (at >= 0 && S.filters[at].v === v) S.filters.splice(at, 1);
    else if (at >= 0) S.filters[at].v = v; else S.filters.push({ q, v });
    renderResults(); await loadItems(true);
  });

  $("#filters").addEventListener("click", async (e) => {
    const b = e.target.closest("[data-unfilter]"); if (!b) return;
    S.filters.splice(+b.dataset.unfilter, 1); renderResults(); await loadItems(true);
  });

  $("#items").addEventListener("click", (e) => {
    if (e.target.closest(".item-detail")) return;
    const li = e.target.closest("[data-item]"); if (!li) return;
    S.openItem = S.openItem === +li.dataset.item ? null : +li.dataset.item; renderItems();
  });
  $("#more").addEventListener("click", () => loadItems(false));

  let searchTimer;
  $("#search").addEventListener("input", (e) => { clearTimeout(searchTimer); searchTimer = setTimeout(() => { S.search = e.target.value; loadItems(true); }, 250); });
  for (const id of PICKERS) {
    const btn = $(id);
    btn.addEventListener("click", () => (P.open && P.anchor === btn ? closePicker(true) : openPicker(btn)));
    btn.addEventListener("keydown", (e) => { if (["Enter", " ", "ArrowDown", "ArrowUp"].includes(e.key)) { e.preventDefault(); openPicker(btn); } });
  }
  $("#mpick-panel").addEventListener("mousedown", (e) => { if (!e.target.closest("button")) e.preventDefault(); });   // keep focus on the listbox
  $("#mpick-panel").addEventListener("click", (e) => {
    if (e.target.closest("a")) { closePicker(false); return; }          // "Compare models": let the link navigate
    const sort = e.target.closest("[data-sort]"), row = e.target.closest(".mpick-row");
    if (sort) { P.sort = sort.dataset.sort; store.set("pickSort", P.sort); P.active = Math.max(0, sortedModels().findIndex((m) => m.name === S.model)); renderPanel(); $("#mpick-panel").focus(); }
    else if (e.target.closest("[data-info]")) { P.info = !P.info; renderPanel(); placePanel(true); $("#mpick-panel [data-info]").focus(); }
    else if (row && !row.classList.contains("dis")) chooseModel(P.list[+row.dataset.i].name);
  });
  $("#mpick-panel").addEventListener("mousemove", (e) => {
    const row = e.target.closest(".mpick-row"); if (!row || row.classList.contains("dis") || +row.dataset.i === P.active) return;
    document.querySelector("#mpick-panel .mpick-row.active")?.classList.remove("active"); row.classList.add("active"); P.active = +row.dataset.i;
    $("#mpick-panel").setAttribute("aria-activedescendant", row.id);
  });
  $("#mpick-panel").addEventListener("keydown", (e) => {
    if (e.key === "Escape") { e.preventDefault(); closePicker(true); return; }
    if (e.key === "Tab") { e.preventDefault(); closePicker(true); return; }
    if (e.target !== $("#mpick-panel")) return;                 // let the sort and info buttons handle their own keys
    if (e.key === "ArrowDown") { e.preventDefault(); moveActive(1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); moveActive(-1); }
    else if (e.key === "Home") { e.preventDefault(); P.active = -1; moveActive(1); }
    else if (e.key === "End") { e.preventDefault(); P.active = P.list.length; moveActive(-1); }
    else if (e.key === "Enter" || e.key === " ") { e.preventDefault(); const m = P.list[P.active]; if (m) chooseModel(m.name); }
  });
  document.addEventListener("mousedown", (e) => {
    if (P.open && !e.target.closest("#mpick-panel") && !e.target.closest(".mpick-btn")) closePicker(false);
  });
  window.addEventListener("resize", () => { if (P.open) placePanel(true); });
  document.addEventListener("scroll", (e) => { if (P.open && !(e.target instanceof Element && e.target.closest("#mpick-panel"))) placePanel(true); }, true);
  $("#scope").addEventListener("change", renderRunbar);
  $("#limit").addEventListener("change", async (e) => { S.limit = +e.target.value; store.set("limit:" + S.pid, S.limit); await loadResults(); await loadItems(true); });

  $("#run").addEventListener("click", async () => {
    try {
      if (S.results.run?.state === "running") { await api(`/api/projects/${S.pid}/run/cancel`, { method: "POST" }); }
      else {
        const out = await api(`/api/projects/${S.pid}/run`, { method: "POST", body: { model: S.model, scope: $("#scope").value, limit: S.limit || null } });
        if (out.message) toast(out.message);
        $("#scope").value = "unsorted";
      }
      await loadResults();
    } catch (err) { toast(err.message); }
  });

  $("#export").addEventListener("click", () => {
    const p = qs(); p.set("filters", JSON.stringify(S.filters));
    window.location = `/api/projects/${S.pid}/export.csv?${p}`;
  });

  window.addEventListener("hashchange", (e) => { if (location.hash.startsWith("#models") && !new URL(e.oldURL).hash.startsWith("#models")) S.cameFromApp = true; }, true);
  boot().then(route).catch((err) => { document.body.insertAdjacentHTML("beforeend", `<div class="toast">Could not reach the local-jev server: ${esc(err.message)}</div>`); });
})();
