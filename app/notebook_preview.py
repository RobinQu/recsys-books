"""Shared nbconvert post-processing for readable embedded previews."""
from __future__ import annotations

PREVIEW_STYLE = """
<style id="recsys-preview-style">
  /* MathJax v3 display math: paper-style centered, larger, scrollable when wide. */
  mjx-container[display="true"], div.math {
    max-width: 100%;
    overflow-x: auto;
    overflow-y: hidden;
    padding: .55rem .25rem;
    text-align: center !important;
    margin: 1.25em auto !important;
  }
  mjx-container {font-size: 115% !important}
  mjx-container mjx-itable {min-width: auto !important}
  .jp-MarkdownOutput pre {white-space:pre-wrap;overflow-wrap:anywhere}
  .jp-Notebook {max-width:1180px;margin:0 auto}
  .jp-Notebook img {max-width:100%;height:auto;border-radius:8px}
  .jp-Notebook figure{margin:1.4em 0}
</style>
"""

# nbconvert's lab template still ships MathJax v2 (2016-era CHTML). Replace the
# whole v2 loader + config block with MathJax v3 tex-chtml: noticeably better
# spacing, centering and font scaling for the formula-heavy tutorial pages.
MATHJAX_V3 = """<!-- Load mathjax -->
<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
    processEscapes: true,
    processEnvironments: true,
    tags: 'ams'
  },
  chtml: {
    scale: 1.18,
    mtextInheritFont: true,
    displayAlign: 'center',
    displayIndent: '0'
  },
  options: {
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
  }
};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script>
<!-- End of mathjax configuration -->"""

_V2_START = "<!-- Load mathjax -->"
_V2_END = "<!-- End of mathjax configuration -->"

# Preview HTML renders inside the detail page's sandboxed iframe. Without
# retargeting, a click on a site-internal link (e.g. a 3.0 curriculum
# cross-reference) loads the full application shell — topbar, sidebar and all —
# inside the iframe, producing nested chrome. Retarget instead:
#   - same-origin links to a different path -> _top (whole page navigates; the
#     detail page keeps the #anchor and focuses it inside its own preview tab)
#   - external links -> _blank
#   - pure #anchor links keep their default in-preview scrolling.
LINK_TARGETS = """<script id="recsys-preview-link-targets">
(function () {
  function retarget() {
    document.querySelectorAll('a[href]').forEach(function (anchor) {
      var href = anchor.getAttribute('href');
      if (!href || href.charAt(0) === '#') return;
      var url;
      try { url = new URL(href, document.baseURI); } catch (error) { return; }
      if (url.origin === location.origin) {
        if (url.pathname !== location.pathname) anchor.target = '_top';
      } else {
        anchor.target = '_blank';
        anchor.rel = 'noopener noreferrer';
      }
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', retarget);
  else retarget();
})();
</script>"""


def _upgrade_mathjax(body: str) -> str:
    start = body.find(_V2_START)
    end = body.find(_V2_END, start)
    if start == -1 or end == -1:
        return body
    return body[:start] + MATHJAX_V3 + body[end + len(_V2_END):]


def polish_preview(body: str) -> str:
    """Upgrade MathJax, retarget iframe links and fix formula sizing/centering."""
    body = _upgrade_mathjax(body)
    body = body.replace("</head>", f"{PREVIEW_STYLE}</head>", 1)
    return body.replace("</body>", f"{LINK_TARGETS}</body>", 1)
