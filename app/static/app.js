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
if (pageLink) setCurrentLink(pageLink);

const sectionLinks = sidebarLinks.filter((link) => {
  const url = new URL(link.href, window.location.href);
  return url.origin === window.location.origin && url.pathname === window.location.pathname && url.hash && $(url.hash);
});
if (sectionLinks.length) {
  const sectionObserver = new IntersectionObserver((entries) => {
    const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    const link = sectionLinks.find((candidate) => candidate.hash === `#${visible.target.id}`);
    if (link) setCurrentLink(link);
  }, { rootMargin: '-15% 0px -70% 0px', threshold: [0, .2, .6] });
  sectionLinks.forEach((link) => sectionObserver.observe($(link.hash)));
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

$$('.chip').forEach((chip) => chip.addEventListener('click', () => {
  $$('.chip').forEach((item) => item.classList.remove('active'));
  chip.classList.add('active');
  $$('.model-card').forEach((card) => {
    card.hidden = chip.dataset.filter !== 'all' && !card.dataset.stage.includes(chip.dataset.filter);
  });
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
    results.innerHTML = hits.length ? hits.map((model) => `<a href="/#model-explorer" data-hit="${model.id}"><b>${model.name}</b><span>${model.stage} · ${model.idea.slice(0, 42)}…</span></a>`).join('') : '<p>未找到匹配模型</p>';
    results.hidden = false;
  });
  results.addEventListener('click', (event) => {
    const link = event.target.closest('[data-hit]');
    if (!link) return;
    results.hidden = true;
    search.value = '';
    setTimeout(() => $(`[data-model="${link.dataset.hit}"]`)?.classList.add('pulse'), 300);
  });
  document.addEventListener('click', (event) => { if (!event.target.closest('.search-wrap')) results.hidden = true; });
}

$$('.timeline-row').forEach((row) => row.addEventListener('click', () => {
  const expanded = row.classList.toggle('expanded');
  row.setAttribute('aria-expanded', String(expanded));
}));
