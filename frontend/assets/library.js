/* Library view: drop manuals, confirm their details, follow processing, browse and remove manuals. */
'use strict';
(() => {
  const { $, el, toast } = Manuora;
  const STAGES = [['upload', 'Upload'], ['parse', 'Parse'], ['chunk', 'Chunk'], ['embed', 'Embed'], ['index', 'Index']];
  const TYPE_LABELS = { user_manual: 'User manual', administration_manual: 'Administration manual', server_manual: 'Server manual',
    developer_manual: 'Developer manual', getting_started_guide: 'Getting started guide', installation_manual: 'Installation manual',
    reference_manual: 'Reference manual', other: 'Other' };
  const root = $('#library');
  let data = { products: [], jobs: [], types: Object.keys(TYPE_LABELS) };
  const pending = [];          // dropped files: {file, state: 'uploading'|'review'|'error', progress, upload, form, error}
  const openLogs = new Set();  // job ids whose log is expanded
  const logs = {};
  let lastStatus = {};
  let timer = null;

  const fmtSize = bytes => bytes > 1e6 ? `${(bytes / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1e3))} KB`;
  const byVersion = (a, b) => String(b.version ?? '').localeCompare(String(a.version ?? ''), undefined, { numeric: true });
  const visible = () => !$('#view-library').hidden;

  // ------------------------------------------------------------------ data
  async function refresh() {
    try {
      const response = await fetch('/api/library');
      if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
      data = await response.json();
    } catch (error) {
      if (visible()) toast(`Library unavailable: ${error.message}`);
      return schedule(15000);
    }
    let finished = false;
    for (const job of data.jobs) {
      if (lastStatus[job.id] && lastStatus[job.id] !== 'done' && job.status === 'done') {
        finished = true; toast(`“${job.title}” is ready to search`);
      }
      if (lastStatus[job.id] === 'running' && job.status === 'failed') toast(`“${job.title}” failed at ${job.stage}`);
      lastStatus[job.id] = job.status;
    }
    if (finished) Manuora.reloadCatalog();
    const active = data.jobs.filter(job => job.status === 'queued' || job.status === 'running').length;
    const badge = $('#jobs-count');
    badge.hidden = !active; badge.textContent = active;
    if (visible()) render();
    schedule(active ? 2000 : visible() ? 10000 : 20000);
  }
  function schedule(ms) { clearTimeout(timer); timer = setTimeout(refresh, ms); }

  // ------------------------------------------------------------------ uploads
  function addFiles(files) {
    for (const file of files) {
      if (!/\.pdf$/i.test(file.name)) { toast(`${file.name} is not a PDF`); continue; }
      const item = { file, state: 'uploading', progress: 0 };
      pending.push(item);
      upload(item);
    }
    render();
  }
  function upload(item) {
    const request = new XMLHttpRequest();
    request.open('POST', `/api/library/uploads?filename=${encodeURIComponent(item.file.name)}`);
    request.setRequestHeader('Content-Type', 'application/pdf');
    request.upload.onprogress = event => { if (event.lengthComputable) { item.progress = event.loaded / event.total; paintUpload(item); } };
    request.onload = () => {
      let body = {};
      try { body = JSON.parse(request.responseText); } catch (_) { /* keep empty */ }
      if (request.status !== 200) { item.state = 'error'; item.error = body.detail || `Upload failed (${request.status})`; }
      else {
        item.state = 'review'; item.upload = body;
        const s = body.suggestion || {};
        item.form = { display_name: s.display_name || '', product_id: s.product_id || null, version: s.version || '',
                      title: s.title || '', type: s.type || 'user_manual' };
      }
      render();
    };
    request.onerror = () => { item.state = 'error'; item.error = 'Network error during upload'; render(); };
    request.send(item.file);
  }
  function paintUpload(item) {
    const bar = item.node?.querySelector('.bar i');
    if (bar) bar.style.width = `${Math.round(item.progress * 100)}%`;
  }
  function drop(item) { pending.splice(pending.indexOf(item), 1); render(); }

  function intent(form, filename) {
    // What submitting this form will do, from the current library (mirrors the server's matching).
    const name = form.display_name.trim().toLowerCase();
    const product = data.products.find(p => p.display_name.toLowerCase() === name || p.product_id === form.product_id && name);
    if (!product) return { kind: 'new', text: name ? `New product: ${form.display_name.trim()}` : 'Choose a product' };
    const version = form.version.trim() || null;
    const release = product.manuals.filter(m => (m.version ?? null) === version);
    if (!release.length) return { kind: 'new', text: version ? `New version of ${product.display_name}: v${version}` : `Adds to ${product.display_name}` };
    const same = release.find(m => m.title.toLowerCase() === form.title.trim().toLowerCase());
    if (same) return { kind: 'replace', text: `Replaces “${same.title}” (v${version ?? '–'})` };
    return { kind: 'add', text: `Adds a manual to ${product.display_name}${version ? ` v${version}` : ''}` };
  }

  async function submit(item, button) {
    button.disabled = true;
    try {
      const response = await fetch('/api/library/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ upload_id: item.upload.upload_id, ...item.form, version: item.form.version || null }) });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || 'Could not start');
      pending.splice(pending.indexOf(item), 1);
      lastStatus[body.id] = body.status;
      await refresh();
    } catch (error) { toast(error.message); button.disabled = false; }
  }
  async function discard(item) {
    if (item.upload) fetch(`/api/library/uploads/${item.upload.upload_id}`, { method: 'DELETE' });
    drop(item);
  }

  // ------------------------------------------------------------------ job actions
  async function act(url, method, okText) {
    const response = await fetch(url, { method });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) toast(body.detail || 'Action failed'); else if (okText) toast(okText);
    refresh();
  }
  async function toggleLog(job) {
    if (openLogs.has(job.id)) openLogs.delete(job.id);
    else { openLogs.add(job.id); logs[job.id] = (await (await fetch(`/api/library/jobs/${job.id}/log`)).json()).log; }
    render();
  }
  function confirmRemove(manual, product) {
    const dialog = $('#confirm');
    $('#confirm-title').textContent = `Remove “${manual.title}”?`;
    $('#confirm-body').textContent = `${product.display_name}${manual.version ? ` v${manual.version}` : ''} will no longer be searched. ` +
      'The PDF and its processed files are moved to data/library/trash, so you can restore them by hand.';
    $('#confirm-ok').onclick = async () => {
      dialog.close(); toast('Removing…');
      await act(`/api/library/documents/${encodeURIComponent(manual.document_id)}`, 'DELETE', `Removed “${manual.title}”`);
      Manuora.reloadCatalog();
    };
    $('#confirm-cancel').onclick = () => dialog.close();
    dialog.showModal();
  }
  function askAbout(job) {
    const product = $('#product'), version = $('#version');
    product.value = job.product_id; product.dispatchEvent(new Event('change'));
    version.value = job.version || ''; version.dispatchEvent(new Event('change'));
    location.hash = '#/chat';
  }

  // ------------------------------------------------------------------ render
  function render() {
    const manuals = data.products.reduce((n, p) => n + p.manuals.length, 0);
    const scroll = $('#view-library').scrollTop;
    root.replaceChildren(...[
      el('div', { className: 'page-head' }, el('h1', { textContent: 'Library' }),
        el('p', { textContent: `${manuals} manuals across ${data.products.length} products. Add a PDF to make it searchable in chat.` })),
      dropzone(),
      pending.length ? el('section', {}, el('h2', { className: 'section-title', textContent: 'New uploads' }),
        el('div', { className: 'jobs' }, ...pending.map(reviewCard))) : null,
      data.jobs.length ? el('section', {}, el('h2', { className: 'section-title', textContent: 'Processing' }),
        el('div', { className: 'jobs' }, ...data.jobs.map(jobCard))) : null,
      el('section', {}, el('h2', { className: 'section-title', textContent: 'Manuals' }),
        el('div', { className: 'products' }, ...data.products.map(productCard)))].filter(Boolean));
    $('#view-library').scrollTop = scroll;
  }

  function dropzone() {
    const input = el('input', { type: 'file', accept: 'application/pdf,.pdf', multiple: true, hidden: true,
      onchange: () => { addFiles([...input.files]); input.value = ''; } });
    const zone = el('div', { className: 'dropzone', tabIndex: 0, role: 'button', 'aria-label': 'Add PDF manuals',
      innerHTML: '<svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/></svg>' },
      el('strong', { textContent: 'Drop PDF manuals here' }),
      el('span', { textContent: 'or click to browse. New products, new versions, or a replacement for an existing manual.' }), input);
    zone.addEventListener('click', () => input.click());
    zone.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); input.click(); } });
    return zone;
  }

  function reviewCard(item) {
    const head = el('div', { className: 'review-head' }, el('div', { className: 'file-icon', textContent: 'PDF' }),
      el('div', { style: 'min-width:0' }, el('div', { className: 'name', textContent: item.file.name }),
        el('div', { className: 'size', textContent: fmtSize(item.file.size) })));
    let body;
    if (item.state === 'uploading') {
      body = el('div', { className: 'step active' }, el('div', { className: 'label' }, el('span', { textContent: 'Uploading…' })),
        el('div', { className: 'bar' }, el('i', { style: `width:${Math.round(item.progress * 100)}%` })));
    } else if (item.state === 'error') {
      body = el('div', { className: 'review-foot' }, el('span', { className: 'badge error', textContent: item.error }),
        el('button', { className: 'btn', type: 'button', textContent: 'Dismiss', onclick: () => drop(item) }));
    } else {
      const form = item.form;
      const products = el('datalist', { id: `products-${item.upload.upload_id}` }, ...data.products.map(p => new Option(p.display_name)));
      const productForName = () => data.products.find(p => p.display_name.toLowerCase() === form.display_name.trim().toLowerCase());
      const versions = el('datalist', { id: `versions-${item.upload.upload_id}` });
      const fillVersions = () => versions.replaceChildren(...unique((productForName()?.manuals || []).map(m => m.version)).map(v => new Option(v)));
      const badge = el('span', { className: 'badge' });
      const paintIntent = () => { const i = intent(form, item.file.name); badge.className = `badge ${i.kind === 'replace' ? 'replace' : 'new'}`; badge.textContent = i.text; };
      const bind = (key, input) => { input.value = form[key] ?? ''; input.addEventListener('input', () => {
        form[key] = input.value; if (key === 'display_name') { form.product_id = productForName()?.product_id || null; fillVersions(); } paintIntent(); }); return input; };
      const type = bind('type', el('select', {}, ...data.types.map(t => new Option(TYPE_LABELS[t] || t, t))));
      type.addEventListener('change', () => { form.type = type.value; });
      fillVersions(); paintIntent();
      const start = el('button', { className: 'btn primary', type: 'button', textContent: 'Add to library' });
      start.addEventListener('click', () => {
        if (!form.display_name.trim() || !form.title.trim()) { toast('Product and title are required'); return; }
        submit(item, start);
      });
      body = el('div', { style: 'display:grid;gap:14px' },
        el('div', { className: 'review-grid' },
          el('label', { className: 'field' }, 'Product', bind('display_name', el('input', { placeholder: 'e.g. KiCad', autocomplete: 'off' })), products),
          el('label', { className: 'field' }, 'Version', (() => { const i = bind('version', el('input', { placeholder: 'optional', autocomplete: 'off' })); i.setAttribute('list', versions.id); return i; })(), versions),
          el('label', { className: 'field' }, 'Manual title', bind('title', el('input', { autocomplete: 'off' }))),
          el('label', { className: 'field' }, 'Type', type)),
        el('div', { className: 'review-foot' }, badge,
          el('button', { className: 'btn ghost', type: 'button', textContent: 'Discard', onclick: () => discard(item) }), start));
      body.querySelector('input').setAttribute('list', products.id);
    }
    const card = el('article', { className: 'card review' }, head, body);
    item.node = card;
    return card;
  }
  const unique = values => [...new Set(values.filter(v => v !== null && v !== undefined))].sort((a, b) => String(b).localeCompare(String(a), undefined, { numeric: true }));

  function jobCard(job) {
    const index = STAGES.findIndex(([key]) => key === job.stage);
    const steps = el('div', { className: 'steps' }, ...STAGES.map(([key, label], i) => {
      let state = '';
      if (job.status === 'done' || i < index) state = 'done';
      else if (i === index) state = job.status === 'failed' ? 'failed' : 'active';
      const determinate = state === 'active' && (key === 'embed' || key === 'index') && job.progress > 0;
      const timing = job.timings?.[key];
      const note = state === 'done' && timing !== undefined ? `${timing < 60 ? Math.round(timing) + ' s' : Math.round(timing / 60) + ' min'}`
        : state === 'active' ? (job.status === 'queued' ? 'Waiting' : (job.detail || '').replace(/^.*· /, '')) : '';
      return el('div', { className: `step ${state}${state === 'active' && !determinate && job.status === 'running' ? ' indeterminate' : ''}` },
        el('div', { className: 'label' }, el('span', { textContent: label }), el('small', { textContent: note })),
        el('div', { className: 'bar' }, el('i', { style: determinate ? `width:${Math.round(job.progress * 100)}%` : '' })));
    }));
    const status = job.status === 'done' ? el('span', { className: 'badge ok', textContent: 'Ready' })
      : job.status === 'failed' ? el('span', { className: 'badge error', textContent: 'Failed' })
      : job.status === 'queued' ? el('span', { className: 'badge', textContent: 'Queued' })
      : el('span', { className: 'spin', 'aria-label': 'Processing' });
    const actions = el('div', { className: 'actions' },
      job.status === 'done' && data.products.some(p => p.manuals.some(m => m.document_id === job.document_id && m.indexed))
        ? el('button', { className: 'btn', type: 'button', textContent: 'Ask about it', onclick: () => askAbout(job) }) : null,
      job.status === 'failed' ? el('button', { className: 'btn primary', type: 'button', textContent: 'Retry', onclick: () => act(`/api/library/jobs/${job.id}/retry`, 'POST') }) : null,
      el('button', { className: 'btn ghost', type: 'button', textContent: openLogs.has(job.id) ? 'Hide log' : 'Log', onclick: () => toggleLog(job) }),
      job.status !== 'running' ? el('button', { className: 'btn ghost', type: 'button', textContent: job.status === 'queued' ? 'Cancel' : 'Dismiss',
        onclick: () => act(`/api/library/jobs/${job.id}`, 'DELETE') }) : null);
    const what = [job.display_name, job.version ? `v${job.version}` : null, job.replaces ? 'replacement' : null].filter(Boolean).join(' · ');
    return el('article', { className: 'card job' },
      el('div', { className: 'job-head' }, status,
        el('div', { style: 'min-width:0' }, el('div', { className: 'name', textContent: job.title }),
          el('div', { className: 'sub', textContent: `${what} · ${job.filename} · ${fmtSize(job.size)}` })), actions),
      steps,
      job.status === 'running' && job.detail ? el('div', { className: 'sub', style: 'font-size:12px;color:var(--muted)', textContent: job.detail }) : null,
      job.status === 'failed' && job.error ? el('div', { className: 'job-error', textContent: job.error }) : null,
      openLogs.has(job.id) ? el('pre', { className: 'log', textContent: logs[job.id] || '(no output yet)' }) : null);
  }

  function productCard(product) {
    const manuals = [...product.manuals].sort(byVersion);
    const versions = new Set(manuals.map(m => m.version)).size;
    return el('article', { className: 'card product' },
      el('div', { className: 'product-head' }, el('h3', { textContent: product.display_name }),
        el('span', { textContent: `${manuals.length} manual${manuals.length > 1 ? 's' : ''} · ${versions} version${versions > 1 ? 's' : ''}` })),
      ...manuals.map(manual => el('div', { className: 'doc-row' },
        el('span', { className: 'ver', textContent: manual.version ? `v${manual.version}` : '—' }),
        el('span', { className: 'doc-title', title: manual.document_id }, manual.title, el('small', { textContent: TYPE_LABELS[manual.type] || manual.type })),
        manual.indexed ? el('span', { className: 'chunks', textContent: `${manual.chunks.toLocaleString()} passages` })
                       : el('span', { className: 'badge replace', textContent: 'Not indexed' }),
        el('button', { className: 'btn danger', type: 'button', textContent: 'Remove', onclick: () => confirmRemove(manual, product) }))));
  }

  // ------------------------------------------------------------------ drag anywhere on the library page
  const view = $('#view-library');
  view.addEventListener('dragover', event => { event.preventDefault(); root.querySelector('.dropzone')?.classList.add('over'); });
  view.addEventListener('dragleave', event => { if (!view.contains(event.relatedTarget)) root.querySelector('.dropzone')?.classList.remove('over'); });
  view.addEventListener('drop', event => {
    event.preventDefault(); root.querySelector('.dropzone')?.classList.remove('over');
    addFiles([...event.dataTransfer.files]);
  });

  Manuora.onRoute.library = () => { render(); refresh(); };
  refresh();
})();
