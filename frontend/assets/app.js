/* Manuora shell: routing, model controls, chat. The Library view lives in library.js. */
'use strict';

const $ = selector => document.querySelector(selector);
const el = (tag, props = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key.includes('-')) node.setAttribute(key, value); else node[key] = value;  // aria-*, data-*
  }
  node.append(...children.filter(child => child !== null && child !== undefined && child !== false));
  return node;
};
window.Manuora = { $, el, catalog: { documents: [], products: {}, titles: {} }, onRoute: {} };

// ---------------------------------------------------------------- routing + mobile menu
const app = $('#app');
const views = { chat: $('#view-chat'), library: $('#view-library') };
function route() {
  const name = location.hash.replace(/^#\//, '') in views ? location.hash.replace(/^#\//, '') : 'chat';
  for (const [key, view] of Object.entries(views)) view.hidden = key !== name;
  document.querySelectorAll('[data-route]').forEach(link => {
    if (link.dataset.route === name) link.setAttribute('aria-current', 'page'); else link.removeAttribute('aria-current');
  });
  closeMenu();
  if (name !== 'chat') closeViewer();
  Manuora.onRoute[name]?.();
  if (name === 'chat') question.focus({ preventScroll: true });
}
const scrim = $('#scrim');
function openMenu() { app.classList.add('menu-open'); scrim.hidden = false; }
function closeMenu() { app.classList.remove('menu-open'); scrim.hidden = true; }
document.querySelectorAll('#menu, [data-menu]').forEach(button => button.addEventListener('click', openMenu));
scrim.addEventListener('click', closeMenu);
window.addEventListener('hashchange', route);

// ---------------------------------------------------------------- model card
const modelDot = $('#model-dot'), modelName = $('#model-name'), modelState = $('#model-state');
const modelLoad = $('#model-load'), modelUnload = $('#model-unload');
let modelBusy = false, chatBusy = false, modelLoaded = false;
function prettyModel(id) {
  const name = (id || '').split('/').pop().split(':')[0].replace(/-gguf$/i, '');
  return name.replace(/-(\d+(?:\.\d+)?)b$/i, ' $1B').replace(/^qwen/i, 'Qwen') || 'Generation model';
}
function paintModel(text) {
  modelState.textContent = text;
  modelDot.className = 'dot' + (modelBusy ? ' busy' : modelLoaded ? ' on' : '');
  modelLoad.disabled = modelBusy || chatBusy || modelLoaded;
  modelUnload.disabled = modelBusy || chatBusy || !modelLoaded;
}
async function refreshModel() {
  if (modelBusy || chatBusy) return;
  try {
    const response = await fetch('/api/model/status');
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'unavailable');
    modelLoaded = data.loaded;
    modelName.textContent = prettyModel(data.model);
    modelName.title = data.model;
    paintModel(modelLoaded ? 'Loaded on GPU' : 'Not loaded · loads on first question');
  } catch (error) {
    modelLoaded = false; paintModel(`Status unavailable (${error.message})`);
    modelLoad.disabled = modelUnload.disabled = true;
  }
}
async function controlModel(action) {
  modelBusy = true; paintModel(action === 'load' ? 'Loading…' : 'Unloading…');
  try {
    const response = await fetch(`/api/model/${action}`, { method: 'POST' });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'failed');
    modelLoaded = data.loaded;
  } catch (error) { toast(`Model ${action} failed: ${error.message}`); }
  finally { modelBusy = false; paintModel(modelLoaded ? 'Loaded on GPU' : 'Not loaded · loads on first question'); }
}
modelLoad.addEventListener('click', () => controlModel('load'));
modelUnload.addEventListener('click', () => controlModel('unload'));
refreshModel();
setInterval(refreshModel, 15000);

function toast(text) {
  const note = el('div', { className: 'toast', role: 'status', textContent: text });
  document.body.append(note);
  setTimeout(() => note.remove(), 3500);
}
Manuora.toast = toast;

// ---------------------------------------------------------------- scope chips
const product = $('#product'), version = $('#version');
const unique = values => [...new Set(values.filter(Boolean))].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
const productName = id => Manuora.catalog.products[id] || id;
const docTitle = id => Manuora.catalog.titles[id] || id;
function fill(select, values, allLabel, label = value => value) {
  const previous = select.value;
  select.replaceChildren(new Option(allLabel, ''), ...values.map(value => new Option(label(value), value)));
  if (values.includes(previous)) select.value = previous;
  select.parentElement.classList.toggle('set', Boolean(select.value));
}
function updateScope() {
  const docs = Manuora.catalog.documents;
  // Versions only mean something within one product: hide the chip until a product is chosen,
  // and when that product has a single version there is nothing to choose.
  // Newest first.
  const versions = product.value ? unique(docs.filter(d => d.product_id === product.value).map(d => d.product_version)).reverse() : [];
  fill(version, versions.length > 1 ? versions : [], 'Any version', value => `v${value}`);
  version.parentElement.hidden = versions.length < 2;
  product.parentElement.classList.toggle('set', Boolean(product.value));
}
[product, version].forEach(select => select.addEventListener('change', updateScope));
async function loadCatalog() {
  try {
    const data = await (await fetch('/api/options')).json();
    Object.assign(Manuora.catalog, { documents: data.documents, products: data.products || {}, titles: data.titles || {} });
    fill(product, unique(data.documents.map(d => d.product_id)), 'All products', productName);
    updateScope();
    if (conversation.exchanges.length) render(); else paintWelcome();
  } catch (_) { status.textContent = 'Could not load the manual list.'; }
}
Manuora.reloadCatalog = loadCatalog;

// ---------------------------------------------------------------- chat
const chatView = $('#view-chat'), thread = $('#thread'), column = $('#column');
const question = $('#question'), send = $('#send'), status = $('#status'), toBottom = $('#to-bottom');
const DEFAULT_HINT = status.textContent;
const conversationKey = 'manuora.conversation.v1';
const fresh = () => ({ id: crypto.randomUUID(), exchanges: [] });
function restore() {
  try {
    const saved = JSON.parse(sessionStorage.getItem(conversationKey));
    if (/^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$/i.test(saved?.id) && Array.isArray(saved.exchanges) &&
        saved.exchanges.every(e => typeof e.question === 'string' && typeof e.answer === 'string' && Array.isArray(e.sources))) return saved;
  } catch (_) { /* no storage: start fresh */ }
  return fresh();
}
let conversation = restore();
function save() {
  try { sessionStorage.setItem(conversationKey, JSON.stringify(conversation)); }
  catch (_) { status.textContent = 'Browser storage is unavailable; a refresh may lose this conversation.'; }
}

function greeting() {
  const hour = new Date().getHours();
  return hour < 5 ? 'Working late?' : hour < 12 ? 'Good morning' : hour < 18 ? 'Good afternoon' : 'Good evening';
}
function paintWelcome() {
  const welcome = column.querySelector('.welcome');
  if (!welcome) return;
  const docs = Manuora.catalog.documents;
  const products = unique(docs.map(d => d.product_id)).length;
  welcome.querySelector('p').replaceChildren(
    docs.length ? `${docs.length} manuals across ${products} products are ready to search. ` : 'Add a manual to get started. ',
    el('a', { href: '#/library', textContent: docs.length ? 'Manage library' : 'Open library' }));
}
function showWelcome() {
  column.replaceChildren(el('div', { className: 'welcome' },
    el('h1', {}, `${greeting()}`, el('span', { className: 'mark', textContent: '.' })),
    el('p', { textContent: 'Ask a question about your product manuals.' })));
  chatView.classList.add('chat-empty');
  paintWelcome();
}
const nearBottom = () => thread.scrollHeight - thread.scrollTop - thread.clientHeight < 80;
const updateBottom = () => { toBottom.hidden = nearBottom(); };
thread.addEventListener('scroll', updateBottom);
toBottom.addEventListener('click', () => thread.scrollTo({ top: thread.scrollHeight, behavior: 'smooth' }));
function append(node, force = false) {
  const stick = force || nearBottom();
  column.querySelector('.welcome')?.remove();
  chatView.classList.remove('chat-empty');
  column.append(node);
  if (stick) thread.scrollTop = thread.scrollHeight;
  updateBottom();
  return node;
}
const avatar = () => el('div', { className: 'avatar', innerHTML:
  '<svg width="16" height="16" viewBox="0 0 32 32" aria-hidden="true"><path d="M7 24V9l9 8 9-8v15" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>' });

function userMessage(text, force) { return append(el('div', { className: 'msg-user', textContent: text }), force); }
// ---------------------------------------------------------------- answer formatting
// A small Markdown subset (headings, lists, code, bold/italic/code spans) built as DOM nodes, never
// innerHTML, so text from a manual cannot inject markup. [n] citations become buttons that jump to source n.
function inline(text, sourceCards) {
  const out = [];
  const pattern = /(`[^`\n]+`|\*\*[^*\n]+\*\*|__[^_\n]+__|(?<![\w*])\*[^*\s][^*\n]*\*(?!\w)|(?<!\w)_[^_\s][^_\n]*_(?!\w)|\[\d+(?:\s*,\s*\d+)*\])/g;
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    if (match.index > last) out.push(text.slice(last, match.index));
    const token = match[0];
    if (token[0] === '`') out.push(el('code', { textContent: token.slice(1, -1) }));
    else if (token.startsWith('**') || token.startsWith('__')) out.push(el('strong', {}, ...inline(token.slice(2, -2), sourceCards)));
    else if (token[0] === '*' || token[0] === '_') out.push(el('em', {}, ...inline(token.slice(1, -1), sourceCards)));
    else out.push(...citations(token, sourceCards));
    last = match.index + token.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}
function citations(token, sourceCards) {
  // "[2]" or "[1, 3]": a button per number that has a source; numbers without one stay as text.
  const numbers = token.slice(1, -1).split(',').map(n => Number(n.trim()));
  if (!numbers.some(n => sourceCards.has(n))) return [token];
  return numbers.map(n => {
    const card = sourceCards.get(n);
    return card ? el('button', { type: 'button', className: 'cite', textContent: String(n), title: 'Show source',
      onclick: () => { card.open = true; card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                       card.classList.add('flash'); setTimeout(() => card.classList.remove('flash'), 1200); } })
      : `[${n}]`;
  });
}
function answerBody(text, sourceCards) {
  const body = el('div', { className: 'answer' });
  const lines = text.replace(/\r\n?/g, '\n').split('\n');
  let paragraph = [], list = null;
  const flushParagraph = () => {
    if (!paragraph.length) return;
    const p = el('p');
    paragraph.forEach((line, i) => { if (i) p.append(el('br')); p.append(...inline(line, sourceCards)); });
    body.append(p); paragraph = [];
  };
  const closeList = () => { list = null; };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const fence = line.match(/^\s*```/);
    if (fence) {
      flushParagraph(); closeList();
      const code = [];
      while (++i < lines.length && !/^\s*```/.test(lines[i])) code.push(lines[i]);
      body.append(el('pre', {}, el('code', { textContent: code.join('\n') })));
      continue;
    }
    const heading = line.match(/^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$/);
    const item = line.match(/^(\s*)(?:([-*+•])|(\d+)[.)])\s+(.*)$/);
    if (!line.trim()) { flushParagraph(); closeList(); }
    else if (heading) {
      flushParagraph(); closeList();
      body.append(el(heading[1].length <= 2 ? 'h3' : 'h4', {}, ...inline(heading[2], sourceCards)));
    } else if (item) {
      flushParagraph();
      const ordered = Boolean(item[3]), nested = item[1].length >= 2 && list;
      const tag = ordered ? 'ol' : 'ul';
      let target = list;
      if (nested) {
        // One level of nesting: an indented item goes into a sub-list of the previous item.
        const parent = list.lastElementChild;
        target = parent.lastElementChild?.tagName === tag.toUpperCase() ? parent.lastElementChild : parent.appendChild(el(tag));
      } else if (!list || list.tagName !== tag.toUpperCase()) {
        list = el(tag);
        if (ordered && item[3] !== '1') list.start = Number(item[3]);
        body.append(list); target = list;
      }
      target.append(el('li', {}, ...inline(item[4], sourceCards)));
    } else if (list && /^\s{2,}\S/.test(line)) {
      // A wrapped continuation of the last list item.
      list.lastElementChild.append(' ', ...inline(line.trim(), sourceCards));
    } else { closeList(); paragraph.push(line); }
  }
  flushParagraph();
  return body;
}
// ---------------------------------------------------------------- source page viewer
// Wide screens show the cited PDF page beside the chat; narrower ones keep opening it in a new tab.
const viewer = $('#viewer'), viewerFrame = $('#viewer-frame');
const wideEnough = window.matchMedia('(min-width: 1100px)');
let viewingCard = null;
function openViewer(url, title, meta, card) {
  viewer.hidden = false; app.classList.add('viewing');
  $('#viewer-title').textContent = title; $('#viewer-meta').textContent = meta; $('#viewer-tab').href = url;
  // Assign through about:blank so a new page in the same PDF still navigates the built-in viewer.
  if (viewerFrame.src.split('#')[0] === new URL(url, location.href).href.split('#')[0]) viewerFrame.src = 'about:blank';
  viewerFrame.src = `${url}&view=FitH&navpanes=0`;  // narrow panel: fit the width, no thumbnail sidebar
  viewingCard?.classList.remove('viewing');
  viewingCard = card; card.classList.add('viewing');
}
function closeViewer() {
  if (viewer.hidden) return;
  viewer.hidden = true; app.classList.remove('viewing'); viewerFrame.src = 'about:blank';
  viewingCard?.classList.remove('viewing'); viewingCard = null;
}
$('#viewer-close').addEventListener('click', closeViewer);
document.addEventListener('keydown', event => { if (event.key === 'Escape' && !$('#confirm').open) closeViewer(); });
wideEnough.addEventListener('change', () => { if (!wideEnough.matches) closeViewer(); });

function sourceCard(source) {
  const pages = source.pages?.length ? `p. ${source.pages.join(', ')}` : '';
  const versionLabel = source.product_version ? ` v${source.product_version}` : '';
  const params = new URLSearchParams({ product_id: source.product_id, product_version: source.product_version || '', document_id: source.document_id });
  const page = source.pages?.[0] || 1, url = `/api/source?${params}#page=${page}`;
  const meta = `${productName(source.product_id)}${versionLabel} · ${pages}`;
  const card = el('details', { className: 'source' },
    el('summary', {},
      el('span', { className: 'num', textContent: `[${source.citation}]` }),
      el('span', { className: 'title', textContent: docTitle(source.document_id) }),
      el('span', { className: 'meta', textContent: meta })),
    source.headings?.length ? el('p', { className: 'path', textContent: source.headings.join(' › ') }) : null,
    el('p', { className: 'excerpt', textContent: source.preview || 'Passage kept from the earlier answer.' }),
    el('a', { href: url, target: '_blank', rel: 'noopener', textContent: `Show page ${page}`,
      onclick: event => {
        if (!wideEnough.matches || event.ctrlKey || event.metaKey || event.shiftKey || event.button) return;  // new tab as usual
        event.preventDefault();
        openViewer(url, docTitle(source.document_id), meta, card);
      } }));
  return card;
}
function scopeLabel(scope) {
  // What the manuals were searched with, e.g. "KiCad v9.0", "KiCad v8.0 and v9.0", "KiCad, all versions".
  const products = scope.products || [], versions = scope.versions || [];
  if (!products.length) return 'all manuals';
  const names = products.map(productName);
  const list = items => items.length < 3 ? items.join(' and ') : `${items.slice(0, -1).join(', ')} and ${items.at(-1)}`;
  if (products.length > 1) return list(names);
  return versions.length ? `${names[0]} ${list(versions.map(v => `v${v}`))}` : `${names[0]}, all versions`;
}
function assistantMessage(text, sources = [], notice = false, scope = null) {
  const cards = new Map(sources.map(source => [source.citation, sourceCard(source)]));
  const content = el('div', {}, answerBody(text, cards));
  if (notice) content.firstChild.classList.add('notice');
  if (scope) content.append(el('p', { className: 'searched', textContent: `Searched ${scopeLabel(scope)}` }));
  if (cards.size) content.append(el('div', { className: 'sources' },
    el('span', { className: 'sources-label', textContent: `${cards.size} source${cards.size > 1 ? 's' : ''}` }), ...cards.values()));
  return append(el('div', { className: 'msg-assistant' }, avatar(), content));
}

function render() {
  if (!conversation.exchanges.length) { showWelcome(); return; }
  column.replaceChildren();
  for (const exchange of conversation.exchanges) {
    userMessage(exchange.question);
    assistantMessage(exchange.answer, exchange.sources, false, exchange.scope);
  }
  thread.scrollTop = thread.scrollHeight;
  updateBottom();
}

const SEND_ICON = send.innerHTML;
const STOP_ICON = '<svg width="14" height="14" viewBox="0 0 24 24" aria-hidden="true"><rect x="5" y="5" width="14" height="14" rx="2.5" fill="currentColor"/></svg>';
function autosize() {
  question.style.height = 'auto';
  question.style.height = `${Math.min(question.scrollHeight, 220)}px`;
  // While a turn runs the send button is a Stop button.
  send.disabled = chatBusy ? !inFlight : !question.value.trim();
  send.innerHTML = chatBusy ? STOP_ICON : SEND_ICON;
  send.classList.toggle('stop', chatBusy);
  send.setAttribute('aria-label', chatBusy ? 'Stop' : 'Send');
}
question.addEventListener('input', autosize);
question.addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); $('#chat-form').requestSubmit(); }
});

$('#new-chat').addEventListener('click', () => {
  if (chatBusy) return;
  conversation = fresh(); save();
  closeViewer();
  question.value = ''; autosize();
  status.textContent = DEFAULT_HINT;
  location.hash = '#/chat';
  showWelcome(); question.focus();
});

// ---------------------------------------------------------------- asking
// /api/chat/stream sends {"stage"} lines while the turn runs, then the checked {"reply"}. Only stages
// stream: an answer is shown once it has passed the citation checks, never as a draft.
const STAGES = [['route', 'Understanding'], ['search', 'Searching'], ['rerank', 'Ranking'], ['write', 'Writing'], ['check', 'Checking']];
let inFlight = null;  // AbortController of the running turn
function progressView() {
  const steps = new Map(STAGES.map(([key, label]) => [key, el('li', { textContent: label })]));
  const clock = el('span', { className: 'elapsed' });
  const view = el('div', { className: 'progress', role: 'status', 'aria-live': 'polite' },
    el('ol', {}, ...steps.values()), clock);
  const started = performance.now();
  const timer = setInterval(() => { clock.textContent = `${Math.round((performance.now() - started) / 1000)}s`; }, 1000);
  return {
    view,
    stage(key) {
      let reached = false;
      for (const [name, item] of [...steps].reverse()) {
        if (name === key) { reached = true; item.className = 'active'; }
        else item.className = reached ? 'done' : '';
      }
      if (key === 'write' && !modelLoaded) steps.get('write').textContent = 'Loading model · Writing';
    },
    stop() { clearInterval(timer); },
  };
}
async function* events(response) {
  // NDJSON: one JSON object per line.
  const reader = response.body.getReader(), decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split('\n'); buffer = lines.pop();
    for (const line of lines) if (line.trim()) yield JSON.parse(line);
    if (done) { if (buffer.trim()) yield JSON.parse(buffer); return; }
  }
}
send.addEventListener('click', event => {
  if (!chatBusy) return;
  event.preventDefault();
  inFlight?.abort();
});

$('#chat-form').addEventListener('submit', async event => {
  event.preventDefault();
  const text = question.value.trim();
  if (!text || chatBusy) return;
  const asked = userMessage(text, true);
  question.value = ''; chatBusy = true; inFlight = new AbortController(); autosize(); paintModel(modelState.textContent);
  $('#new-chat').disabled = true;
  const progress = progressView();
  const pending = append(el('div', { className: 'msg-assistant' }, avatar(), progress.view), true);
  progress.stage('route');
  try {
    const response = await fetch('/api/chat/stream', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      signal: inFlight.signal,
      body: JSON.stringify({ question: text, product_id: product.value || null, product_version: version.value || null,
                             conversation_id: conversation.id }) });
    if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || `Request failed (${response.status})`);
    let data = null;
    for await (const event of events(response)) {
      if (event.stage) progress.stage(event.stage);
      else if (event.reply) data = event.reply;
      else if (event.error) throw new Error(event.error);
    }
    if (!data) throw new Error('The server ended the reply early');
    pending.remove();
    assistantMessage(data.answer, data.sources, !data.sources.length, data.scope);
    conversation.id = data.conversation_id;
    conversation.exchanges.push({ question: text, answer: data.answer, sources: data.sources, scope: data.scope || null });
    save();
    status.textContent = DEFAULT_HINT;
  } catch (error) {
    pending.remove();
    if (error.name === 'AbortError') {
      // Stopped: the server records nothing, so drop the question from the thread and hand it back for editing.
      asked.remove();
      if (!column.children.length) showWelcome();
      question.value = text;
      status.textContent = 'Stopped. Your question is back in the box. A reply already being written may keep the GPU busy for a moment.';
    } else assistantMessage(`Sorry, that request failed: ${error.message}`, [], true);
  } finally {
    progress.stop();
    chatBusy = false; inFlight = null; autosize(); $('#new-chat').disabled = false; refreshModel(); question.focus();
  }
});

save();
render();
autosize();
loadCatalog();
route();
