"""Build report.pdf from report.html (two passes: the second fills TOC page numbers).

Usage (from this folder):  python3 build.py
Needs google-chrome and poppler-utils (pdftotext).
"""
import html, re, subprocess, pathlib, unicodedata

here = pathlib.Path(__file__).resolve().parent
src = (here / 'report.html').read_text(encoding='utf-8')
heads = [(int(l), re.sub(r'<[^>]+>', '', t).strip())
         for l, t in re.findall(r'<h[12][^>]*data-toc="(\d)"[^>]*>(.*?)</h[12]>', src)]

def toc(pages):
    rows, ch = [], 0
    for (lvl, text), pg in zip(heads, pages):
        label = text
        if lvl == 1 and not text.startswith(('References', 'Appendix')):
            ch += 1
            label = f'{ch}&nbsp;&nbsp;{text}'
        rows.append(f'<div class="r l{lvl}"><span class="t">{label}</span><span class="d"></span><span class="p">{pg}</span></div>')
    return '\n'.join(rows)

def render(pages, out):
    tmp = here / '_build.html'
    tmp.write_text(src.replace('<!--TOC-->', toc(pages)), encoding='utf-8')
    subprocess.run(['google-chrome', '--headless=new', '--no-sandbox', '--allow-file-access-from-files',
                    '--virtual-time-budget=15000', '--no-pdf-header-footer',
                    f'--print-to-pdf={out}', tmp.as_uri()], check=True, capture_output=True)
    tmp.unlink()

def norm(s):
    s = unicodedata.normalize('NFKC', html.unescape(s))
    return re.sub(r'[^a-z0-9]+', '', s.lower())

def find_pages(pdf):
    n = int(re.search(r'Pages:\s+(\d+)', subprocess.run(['pdfinfo', str(pdf)], capture_output=True, text=True).stdout).group(1))
    texts = [norm(subprocess.run(['pdftotext', '-f', str(i), '-l', str(i), str(pdf), '-'],
                                 capture_output=True, text=True).stdout) for i in range(1, n + 1)]
    start = next(i for i, t in enumerate(texts) if 'tableofcontents' in t) + 1
    # skip the TOC page(s): body starts at the first page containing chapter 1's heading after the TOC
    while 'chapter1' not in texts[start]:
        start += 1
    pages, cur = [], start
    for _, text in heads:
        key = norm(text)
        while cur < n and key not in texts[cur]:
            cur += 1
        if cur >= n:
            raise SystemExit(f'heading not found: {text}')
        pages.append(cur + 1)
    return pages

pdf = here / 'Thesis_report.pdf'
render(['00'] * len(heads), pdf)
pages = find_pages(pdf)
render(pages, pdf)
assert find_pages(pdf) == pages, 'TOC page numbers moved between passes'
print('pages:', dict(zip([t for _, t in heads], pages)))
