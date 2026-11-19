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
const product = $('#product'), version = $('#version'), documentSelect = $('#document');
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
  fill(version, unique(docs.filter(d => !product.value || d.product_id === product.value).map(d => d.product_version)),
       'Any version', value => `v${value}`);
  fill(documentSelect, unique(docs.filter(d => (!product.value || d.product_id === product.value) &&
       (!version.value || d.product_version === version.value)).map(d => d.document_id)), 'All manuals', docTitle);
  product.parentElement.classList.toggle('set', Boolean(product.value));
}
[product, version, documentSelect].forEach(select => select.addEventListener('change', updateScope));
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
function answerBody(text, sourceCards) {
  // Plain text with [n] citations turned into buttons that jump to source n.
  const body = el('div', { className: 'answer' });
  const parts = text.split(/(\[\d+\])/g);
  for (const part of parts) {
    const match = part.match(/^\[(\d+)\]$/);
    const card = match && sourceCards.get(Number(match[1]));
    if (card) {
      body.append(el('button', { type: 'button', className: 'cite', textContent: match[1], title: 'Show source',
        onclick: () => { card.open = true; card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                         card.classList.add('flash'); setTimeout(() => card.classList.remove('flash'), 1200); } }));
    } else body.append(part);
  }
  return body;
}
function sourceCard(source) {
  const pages = source.pages?.length ? `p. ${source.pages.join(', ')}` : '';
  const versionLabel = source.product_version ? ` v${source.product_version}` : '';
  const params = new URLSearchParams({ product_id: source.product_id, product_version: source.product_version || '', document_id: source.document_id });
  return el('details', { className: 'source' },
    el('summary', {},
      el('span', { className: 'num', textContent: `[${source.citation}]` }),
      el('span', { className: 'title', textContent: docTitle(source.document_id) }),
      el('span', { className: 'meta', textContent: `${productName(source.product_id)}${versionLabel} · ${pages}` })),
    source.headings?.length ? el('p', { className: 'path', textContent: source.headings.join(' › ') }) : null,
    el('p', { className: 'excerpt', textContent: source.preview || 'Passage kept from the earlier answer.' }),
    el('a', { href: `/api/source?${params}#page=${source.pages?.[0] || 1}`, target: '_blank', rel: 'noopener', textContent: 'Open PDF at this page ↗' }));
}
function assistantMessage(text, sources = [], notice = false) {
  const cards = new Map(sources.map(source => [source.citation, sourceCard(source)]));
  const content = el('div', {}, answerBody(text, cards));
  if (notice) content.firstChild.classList.add('notice');
  if (cards.size) content.append(el('div', { className: 'sources' },
    el('span', { className: 'sources-label', textContent: `${cards.size} source${cards.size > 1 ? 's' : ''}` }), ...cards.values()));
  return append(el('div', { className: 'msg-assistant' }, avatar(), content));
}

function render() {
  if (!conversation.exchanges.length) { showWelcome(); return; }
  column.replaceChildren();
  for (const exchange of conversation.exchanges) {
    userMessage(exchange.question);
    assistantMessage(exchange.answer, exchange.sources);
  }
  thread.scrollTop = thread.scrollHeight;
  updateBottom();
}

function autosize() {
  question.style.height = 'auto';
  question.style.height = `${Math.min(question.scrollHeight, 220)}px`;
  send.disabled = chatBusy || !question.value.trim();
}
question.addEventListener('input', autosize);
question.addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); $('#chat-form').requestSubmit(); }
});

$('#new-chat').addEventListener('click', () => {
  if (chatBusy) return;
  conversation = fresh(); save();
  question.value = ''; autosize();
  status.textContent = DEFAULT_HINT;
  location.hash = '#/chat';
  showWelcome(); question.focus();
});

$('#chat-form').addEventListener('submit', async event => {
  event.preventDefault();
  const text = question.value.trim();
  if (!text || chatBusy) return;
  userMessage(text, true);
  question.value = ''; chatBusy = true; autosize(); paintModel(modelState.textContent);
  $('#new-chat').disabled = true;
  const pending = append(el('div', { className: 'msg-assistant' }, avatar(),
    el('div', { className: 'thinking', role: 'status' }, el('span', { className: 'pulse', innerHTML: '<i></i><i></i><i></i>' }),
      el('span', { textContent: modelLoaded ? 'Searching the manuals…' : 'Loading the model and searching the manuals…' }))), true);
  try {
    const response = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: text, product_id: product.value || null, product_version: version.value || null,
                             document_id: documentSelect.value || null, conversation_id: conversation.id }) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Request failed');
    pending.remove();
    assistantMessage(data.answer, data.sources, !data.sources.length);
    conversation.id = data.conversation_id;
    conversation.exchanges.push({ question: text, answer: data.answer, sources: data.sources });
    save();
    status.textContent = DEFAULT_HINT;
  } catch (error) {
    pending.remove();
    assistantMessage(`Sorry, that request failed: ${error.message}`, [], true);
  } finally {
    chatBusy = false; autosize(); $('#new-chat').disabled = false; refreshModel(); question.focus();
  }
});

save();
render();
autosize();
loadCatalog();
route();
