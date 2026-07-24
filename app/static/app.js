const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const modelData = $('#model-data');
const models = modelData ? JSON.parse(modelData.textContent) : [];

const colorScheme = window.matchMedia('(prefers-color-scheme: light)');
const savedTheme = localStorage.getItem('theme');
const applyTheme = (theme) => {
  document.documentElement.dataset.theme = theme;
  const button = $('#theme-toggle');
  if (button) {
    button.setAttribute('aria-label', theme === 'light' ? '切换到深色主题' : '切换到浅色主题');
    button.title = theme === 'light' ? '当前：浅色' : '当前：深色';
  }
};
applyTheme(savedTheme || (colorScheme.matches ? 'light' : 'dark'));
const themeToggle = $('#theme-toggle');
themeToggle?.addEventListener('click', () => {
  const next = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
  applyTheme(next);
  localStorage.setItem('theme', next);
});
colorScheme.addEventListener('change', (event) => {
  if (!localStorage.getItem('theme')) applyTheme(event.matches ? 'light' : 'dark');
});

const revealObserver = new IntersectionObserver((entries) => entries.forEach((entry) => {
  if (entry.isIntersecting) {
    entry.target.classList.add('visible');
    revealObserver.unobserve(entry.target);
  }
}), { threshold: .08 });
$$('.reveal').forEach((element) => revealObserver.observe(element));

const lazyFrames = new IntersectionObserver((entries) => entries.forEach((entry) => {
  if (entry.isIntersecting && entry.target.dataset.src) {
    entry.target.src = entry.target.dataset.src;
    delete entry.target.dataset.src;
    lazyFrames.unobserve(entry.target);
  }
}), { rootMargin: '500px' });
$$('iframe[data-src]').forEach((frame) => lazyFrames.observe(frame));

// Shared sidebar behavior: current location, scroll spy, and mobile drawer.
const sidebar = $('.sidebar');
const sidebarLinks = sidebar ? $$('a', sidebar) : [];
const setCurrentLink = (activeLink) => {
  sidebarLinks.forEach((link) => {
    link.classList.toggle('active', link === activeLink);
    if (link === activeLink) link.setAttribute('aria-current', link.hash ? 'location' : 'page');
    else link.removeAttribute('aria-current');
  });
};

const pageLink = sidebarLinks.find((link) => {
  const url = new URL(link.href, window.location.href);
  return url.origin === window.location.origin && url.pathname === window.location.pathname && !url.hash;
});
if (pageLink) {
  setCurrentLink(pageLink);
  // Layout, fonts and restored scroll state can all settle after the first
  // animation frame. Re-run the same idempotent positioning at each boundary.
  const revealCurrentPage = () => {
    const sidebarRect = sidebar.getBoundingClientRect();
    const linkRect = pageLink.getBoundingClientRect();
    const visibleTop = sidebarRect.top + 18;
    const visibleBottom = sidebarRect.bottom - 18;
    if (linkRect.top < visibleTop || linkRect.bottom > visibleBottom) {
      const centered = sidebar.scrollTop + linkRect.top - sidebarRect.top - (sidebarRect.height - linkRect.height) / 2;
      sidebar.scrollTo({ top: Math.max(0, centered), behavior: 'instant' });
    }
  };
  requestAnimationFrame(() => requestAnimationFrame(revealCurrentPage));
  window.addEventListener('load', revealCurrentPage, { once: true });
  document.fonts?.ready.then(revealCurrentPage);
}

const sectionLinks = sidebarLinks.filter((link) => {
  const url = new URL(link.href, window.location.href);
  return url.origin === window.location.origin && url.pathname === window.location.pathname && url.hash && $(url.hash);
});
if (sectionLinks.length) {
  let sectionUpdateQueued = false;
  const updateCurrentSection = () => {
    sectionUpdateQueued = false;
    const activationLine = window.innerHeight * .24;
    const passed = sectionLinks.filter((link) => $(link.hash).getBoundingClientRect().top <= activationLine);
    setCurrentLink(passed.at(-1) || sectionLinks[0]);
  };
  const queueSectionUpdate = () => {
    if (sectionUpdateQueued) return;
    sectionUpdateQueued = true;
    requestAnimationFrame(updateCurrentSection);
  };
  window.addEventListener('scroll', queueSectionUpdate, { passive: true });
  window.addEventListener('resize', queueSectionUpdate);
  window.addEventListener('hashchange', queueSectionUpdate);
  requestAnimationFrame(() => requestAnimationFrame(updateCurrentSection));
}

const sidebarToggle = $('.sidebar-toggle');
if (sidebar && sidebarToggle) {
  const scrim = document.createElement('div');
  scrim.className = 'sidebar-scrim';
  document.body.appendChild(scrim);
  const setSidebarOpen = (open) => {
    sidebar.classList.toggle('mobile-open', open);
    scrim.classList.toggle('visible', open);
    document.body.classList.toggle('nav-open', open);
    sidebarToggle.setAttribute('aria-expanded', String(open));
    sidebarToggle.setAttribute('aria-label', open ? '关闭章节导航' : '打开章节导航');
  };
  sidebarToggle.addEventListener('click', () => setSidebarOpen(!sidebar.classList.contains('mobile-open')));
  scrim.addEventListener('click', () => setSidebarOpen(false));
  sidebarLinks.forEach((link) => link.addEventListener('click', () => setSidebarOpen(false)));
  window.addEventListener('keydown', (event) => { if (event.key === 'Escape') setSidebarOpen(false); });
  window.addEventListener('resize', () => { if (window.innerWidth > 980) setSidebarOpen(false); });
}

$$('.chip[data-filter]').forEach((chip) => chip.addEventListener('click', () => {
  const filter = chip.dataset.filter;
  $$('.chip[data-filter]').forEach((item) => {
    item.classList.remove('active');
    item.setAttribute('aria-pressed', 'false');
  });
  chip.classList.add('active');
  chip.setAttribute('aria-pressed', 'true');
  const cards = $$('.model-card');
  cards.forEach((card) => {
    card.hidden = filter !== 'all' && !card.dataset.stage.includes(filter);
  });
  const visibleCount = cards.filter((card) => !card.hidden).length;
  const status = $('#model-filter-status');
  if (status) status.textContent = filter === 'all' ? `显示全部 ${visibleCount} 个模型` : `${filter} · ${visibleCount} 个模型`;
  window.dispatchEvent(new CustomEvent('model-filter-change', { detail: { filter } }));
}));

const compareDrawer = $('#compare-drawer');
const selected = new Set();
function renderDrawer() {
  if (!compareDrawer) return;
  compareDrawer.hidden = selected.size === 0;
  $('#compare-count').textContent = `${selected.size}/3`;
  $$('.compare-toggle').forEach((button) => {
    button.classList.toggle('selected', selected.has(button.dataset.compare));
    button.textContent = selected.has(button.dataset.compare) ? '✓ 已选' : '＋ 对比';
  });
}
$$('.compare-toggle').forEach((button) => button.addEventListener('click', () => {
  const id = button.dataset.compare;
  if (selected.has(id)) selected.delete(id);
  else if (selected.size < 3) selected.add(id);
  renderDrawer();
}));
$('#compare-clear')?.addEventListener('click', () => { selected.clear(); renderDrawer(); });
$('#compare-open')?.addEventListener('click', () => {
  const rows = models.filter((model) => selected.has(model.id));
  $('#compare-table').innerHTML = `<div class="compare-table"><div><b>维度</b>${rows.map((model) => `<b>${model.name}</b>`).join('')}</div>${[['阶段','stage'],['核心','idea'],['优势','pros'],['短板','cons'],['指标','metric']].map(([label, key]) => `<div><b>${label}</b>${rows.map((model) => `<span>${model[key]}</span>`).join('')}</div>`).join('')}</div>`;
  $('#compare-dialog').showModal();
});

const search = $('#global-search');
const results = $('#search-results');
if (search && results) {
  search.addEventListener('input', () => {
    const query = search.value.trim().toLowerCase();
    if (!query) { results.hidden = true; return; }
    const hits = models.filter((model) => Object.values(model).join(' ').toLowerCase().includes(query)).slice(0, 6);
    results.innerHTML = hits.length ? hits.map((model) => `<a href="/catalog#model-explorer" data-hit="${model.id}"><b>${model.name}</b><span>${model.stage} · ${model.idea.slice(0, 42)}…</span></a>`).join('') : '<p>未找到匹配模型</p>';
    results.hidden = false;
  });
  results.addEventListener('click', (event) => {
    const link = event.target.closest('[data-hit]');
    if (!link) return;
    results.hidden = true;
    search.value = '';
    sessionStorage.setItem('model-search-hit', link.dataset.hit);
    setTimeout(() => $(`[data-model="${link.dataset.hit}"]`)?.classList.add('pulse'), 300);
  });
  document.addEventListener('click', (event) => { if (!event.target.closest('.search-wrap')) results.hidden = true; });
}

$$('.timeline-row').forEach((row) => row.addEventListener('click', () => {
  const expanded = row.classList.toggle('expanded');
  row.setAttribute('aria-expanded', String(expanded));
}));

// Chapter 2 publishes the graph hooks.  The server owns expansion and
// limits; this renderer only displays a bounded deterministic view and never
// creates an unbounded client-side force graph.
(() => {
  const dataElement = document.querySelector('#knowledge-graph-data');
  const graphRoot = document.querySelector('#knowledge-graph');
  if (!dataElement || !graphRoot) return;

  let payload;
  try {
    payload = JSON.parse(dataElement.textContent);
  } catch (error) {
    graphRoot.hidden = true;
    return;
  }

  const MAX_VISIBLE_NODES = 16;
  const VISUAL_EDGE_TYPES = new Set(['contains', 'emphasizes', 'evolves', 'prerequisite', 'requires']);
  const defaultView = payload.default || payload;
  const views = payload.views || {};
  const filterViews = payload.filters || {};
  let currentView = defaultView;
  let selectedId = null;

  const toolbar = document.createElement('div');
  toolbar.className = 'knowledge-graph-toolbar';
  const selectLabel = document.createElement('label');
  selectLabel.textContent = '聚焦章节';
  selectLabel.htmlFor = 'knowledge-graph-chapter-select';
  const chapterSelect = document.createElement('select');
  chapterSelect.id = 'knowledge-graph-chapter-select';
  const overviewOption = document.createElement('option');
  overviewOption.value = '';
  overviewOption.textContent = '全部章节';
  chapterSelect.appendChild(overviewOption);
  const resetButton = document.createElement('button');
  resetButton.type = 'button';
  resetButton.className = 'ghost knowledge-graph-reset';
  resetButton.textContent = '返回总览';
  toolbar.append(selectLabel, chapterSelect, resetButton);

  const stage = document.createElement('div');
  stage.className = 'knowledge-graph-stage';
  stage.setAttribute('aria-label', '教程知识关系图');
  const edgeLayer = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  edgeLayer.classList.add('knowledge-graph-edges');
  edgeLayer.setAttribute('aria-hidden', 'true');
  const nodeLayer = document.createElement('div');
  nodeLayer.className = 'knowledge-graph-nodes';
  stage.append(edgeLayer, nodeLayer);

  const relations = document.createElement('aside');
  relations.className = 'knowledge-graph-relations';
  relations.setAttribute('aria-live', 'polite');
  relations.setAttribute('aria-atomic', 'true');

  graphRoot.replaceChildren(toolbar, stage, relations);

  const defaultChapters = (defaultView.nodes || []).filter((node) => node.type === 'chapter');
  defaultChapters.forEach((chapter) => {
    const option = document.createElement('option');
    option.value = chapter.id;
    option.textContent = `${chapter.number ? `${chapter.number} · ` : ''}${chapter.label}`;
    chapterSelect.appendChild(option);
  });

  function boundedView(view) {
    const nodes = (view?.nodes || []).slice(0, MAX_VISIBLE_NODES);
    const ids = new Set(nodes.map((node) => node.id));
    const edges = (view?.edges || []).filter((edge) => ids.has(edge.source) && ids.has(edge.target));
    return { ...view, nodes, edges };
  }

  function viewFor(focusId) {
    if (!focusId) return boundedView(defaultView);
    return boundedView(views[focusId] || views[focusId.replace(/^chapter:/, '')] || currentView);
  }

  function nodeColumn(node) {
    if (node.type === 'math') return 1;
    if (node.type === 'chapter') return 2;
    return 3;
  }

  function drawEdges() {
    edgeLayer.replaceChildren();
    const stageRect = stage.getBoundingClientRect();
    edgeLayer.setAttribute('viewBox', `0 0 ${Math.max(1, stageRect.width)} ${Math.max(1, stageRect.height)}`);
    currentView.edges.forEach((edge) => {
      // Keep the picture scannable: detailed model→math dependencies remain
      // available in the accessible text list after selecting either node.
      if (!VISUAL_EDGE_TYPES.has(edge.type)) return;
      if (edge.type === 'emphasizes' && edge.source !== currentView.state?.focus) return;
      const source = nodeLayer.querySelector(`[data-graph-node="${CSS.escape(edge.source)}"]`);
      const target = nodeLayer.querySelector(`[data-graph-node="${CSS.escape(edge.target)}"]`);
      if (!source || !target) return;
      const from = source.getBoundingClientRect();
      const to = target.getBoundingClientRect();
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', String(from.left + from.width / 2 - stageRect.left));
      line.setAttribute('y1', String(from.top + from.height / 2 - stageRect.top));
      line.setAttribute('x2', String(to.left + to.width / 2 - stageRect.left));
      line.setAttribute('y2', String(to.top + to.height / 2 - stageRect.top));
      line.dataset.edgeType = edge.type;
      edgeLayer.appendChild(line);
    });
  }

  function renderRelations() {
    relations.replaceChildren();
    const heading = document.createElement('h4');
    const selected = currentView.nodes.find((node) => node.id === selectedId);
    const activeFilter = currentView.state?.filter;
    heading.textContent = selected
      ? `${selected.label} · 相关关系`
      : activeFilter
        ? `${activeFilter} · ${currentView.state?.model_count || 0} 个模型，${currentView.nodes.length} 个关系节点`
        : `当前显示 ${currentView.nodes.length} 个节点`;
    relations.appendChild(heading);
    if (!selected) {
      const hint = document.createElement('p');
      hint.textContent = '选择章节或节点，查看可朗读的文字关系；图中连线仅作视觉辅助。';
      relations.appendChild(hint);
      return;
    }
    const list = document.createElement('ul');
    currentView.edges.filter((edge) => edge.source === selectedId || edge.target === selectedId).forEach((edge) => {
      const otherId = edge.source === selectedId ? edge.target : edge.source;
      const other = currentView.nodes.find((node) => node.id === otherId);
      if (!other) return;
      const item = document.createElement('li');
      item.textContent = `${edge.label}：${other.label}`;
      list.appendChild(item);
    });
    if (!list.children.length) {
      const item = document.createElement('li');
      item.textContent = '当前视图没有展开更多关系。';
      list.appendChild(item);
    }
    relations.appendChild(list);
    if (selected.url) {
      const link = document.createElement('a');
      link.href = selected.url;
      link.textContent = '打开对应教程 →';
      relations.appendChild(link);
    }
  }

  function selectNode(nodeId) {
    selectedId = nodeId;
    nodeLayer.querySelectorAll('[data-graph-node]').forEach((button) => {
      button.setAttribute('aria-pressed', String(button.dataset.graphNode === selectedId));
    });
    renderRelations();
  }

  function render(view, focusId = null) {
    currentView = boundedView(view);
    selectedId = focusId && currentView.nodes.some((node) => node.id === focusId) ? focusId : null;
    nodeLayer.replaceChildren();
    const rows = new Map();
    currentView.nodes.forEach((node) => {
      const column = nodeColumn(node);
      const row = (rows.get(column) || 0) + 1;
      rows.set(column, row);
      const button = document.createElement('button');
      button.type = 'button';
      button.className = `knowledge-graph-node is-${node.type}`;
      button.dataset.graphNode = node.id;
      button.style.setProperty('--graph-column', String(column));
      button.style.setProperty('--graph-row', String(row));
      button.setAttribute('aria-pressed', String(node.id === selectedId));
      const type = document.createElement('small');
      type.textContent = node.type === 'chapter' ? `章节 ${node.number || ''}` : node.type === 'model' ? node.stage || '算法' : node.area || '数学';
      const label = document.createElement('b');
      label.textContent = node.label;
      button.append(type, label);
      button.addEventListener('click', () => {
        if (node.type === 'chapter' && views[node.id]) {
          chapterSelect.value = node.id;
          render(viewFor(node.id), node.id);
        } else {
          selectNode(node.id);
        }
      });
      nodeLayer.appendChild(button);
    });
    renderRelations();
    requestAnimationFrame(drawEdges);
  }

  function resetGraph() {
    chapterSelect.value = '';
    render(viewFor(null));
  }

  chapterSelect.addEventListener('change', () => {
    const focusId = chapterSelect.value || null;
    render(viewFor(focusId), focusId);
  });
  resetButton.addEventListener('click', resetGraph);
  window.addEventListener('model-filter-change', (event) => {
    const filter = event.detail?.filter;
    chapterSelect.value = '';
    if (!filter || filter === 'all' || !filterViews[filter]) render(viewFor(null));
    else render(filterViews[filter], `filter:${filter}`);
  });
  graphRoot.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      event.preventDefault();
      resetGraph();
      chapterSelect.focus();
    }
  });
  new ResizeObserver(() => requestAnimationFrame(drawEdges)).observe(stage);
  render(viewFor(null));
})();

const pendingModelHit = sessionStorage.getItem('model-search-hit');
if (pendingModelHit) {
  const hit = $(`[data-model="${CSS.escape(pendingModelHit)}"]`);
  if (hit) {
    sessionStorage.removeItem('model-search-hit');
    hit.classList.add('pulse');
    setTimeout(() => hit.classList.remove('pulse'), 1800);
  }
}
