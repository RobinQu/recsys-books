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


def _upgrade_mathjax(body: str) -> str:
    start = body.find(_V2_START)
    end = body.find(_V2_END, start)
    if start == -1 or end == -1:
        return body
    return body[:start] + MATHJAX_V3 + body[end + len(_V2_END):]


def polish_preview(body: str) -> str:
    """Upgrade MathJax and add formula sizing/centering fixes without changing outputs."""
    body = _upgrade_mathjax(body)
    return body.replace("</head>", f"{PREVIEW_STYLE}</head>", 1)
