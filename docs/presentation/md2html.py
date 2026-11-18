"""Minimal Markdown -> styled HTML for QUICK_SUMMARY (headings, lists, tables, quotes)."""
import html, re, sys

src, dst = sys.argv[1], sys.argv[2]
lines = open(src, encoding='utf-8').read().splitlines()

def inline(t):
    t = html.escape(t, quote=False)
    t = re.sub(r'`([^`]+)`', r'<code>\1</code>', t)
    t = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', t)
    t = re.sub(r'(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])', r'<i>\1</i>', t)
    return t

out, i, para = [], 0, []
def flush():
    if para:
        out.append('<p>' + inline(' '.join(para)) + '</p>'); para.clear()

while i < len(lines):
    l = lines[i]
    if not l.strip():
        flush(); i += 1; continue
    m = re.match(r'^(#{1,3}) (.*)', l)
    if m:
        flush(); n = len(m.group(1)); out.append(f'<h{n}>{inline(m.group(2))}</h{n}>'); i += 1; continue
    m = re.match(r'^!\[(.*)\]\((.*)\)$', l.strip())
    if m:
        import os
        flush(); src_img = os.path.abspath(os.path.join(os.path.dirname(src), m.group(2)))
        out.append(f'<img src="file://{src_img}" alt="{html.escape(m.group(1))}" style="width:100%;margin:6px 0;border:1px solid #dde3ec">'); i += 1; continue
    if l.strip() == '---':
        flush(); out.append('<hr>'); i += 1; continue
    if l.startswith('|'):
        flush(); rows = []
        while i < len(lines) and lines[i].startswith('|'):
            rows.append([c.strip() for c in lines[i].strip().strip('|').split('|')]); i += 1
        h, body = rows[0], rows[2:]
        out.append('<table><tr>' + ''.join(f'<th>{inline(c)}</th>' for c in h) + '</tr>' +
                   ''.join('<tr>' + ''.join(f'<td>{inline(c)}</td>' for c in r) + '</tr>' for r in body) + '</table>')
        continue
    if l.startswith('>'):
        flush(); q = []
        while i < len(lines) and lines[i].startswith('>'):
            q.append(lines[i][1:].strip()); i += 1
        out.append('<blockquote>' + inline(' '.join(q)) + '</blockquote>'); continue
    m = re.match(r'^(\s*)([-*]|\d+\.) (.*)', l)
    if m:
        flush()
        # collect a list block (top level + one nested level)
        items = []
        while i < len(lines):
            mm = re.match(r'^(\s*)([-*]|\d+\.) (.*)', lines[i])
            if mm:
                items.append((len(mm.group(1)) > 0, mm.group(2)[0].isdigit(), mm.group(3))); i += 1
            elif lines[i].startswith('  ') and lines[i].strip() and items:
                d, o, t = items[-1]; items[-1] = (d, o, t + ' ' + lines[i].strip()); i += 1
            else:
                break
        tag = 'ol' if items[0][1] else 'ul'
        h = [f'<{tag}>']; open_sub = None
        for nested, ordered, text in items:
            if nested:
                if not open_sub:
                    open_sub = 'ol' if ordered else 'ul'; h[-1] = h[-1].removesuffix('</li>'); h.append(f'<{open_sub}>')
                h.append(f'<li>{inline(text)}</li>')
            else:
                if open_sub:
                    h.append(f'</{open_sub}></li>'); open_sub = None
                h.append(f'<li>{inline(text)}</li>')
        if open_sub:
            h.append(f'</{open_sub}></li>')
        h.append(f'</{tag}>'); out.append(''.join(h)); continue
    para.append(l.strip()); i += 1
flush()

css = """
@page{size:A4;margin:18mm 17mm}
body{font-family:'Segoe UI',Inter,'Liberation Sans',Arial,sans-serif;color:#1d2330;font-size:10.6pt;line-height:1.55}
h1{font-size:19pt;margin:0 0 6px;color:#1B2333}
h2{font-size:13.5pt;margin:20px 0 6px;color:#1B2333;break-after:avoid}
h3{font-size:11.5pt;margin:14px 0 4px;color:#2F5D9E;break-after:avoid}
p{margin:6px 0} ul,ol{margin:6px 0 6px 22px;padding:0} li{margin:3px 0}
hr{border:0;border-top:1px solid #d3dae5;margin:14px 0}
blockquote{margin:8px 0;padding:8px 12px;background:#FBE9E5;border-left:3px solid #B4442F;font-size:10pt}
table{border-collapse:collapse;width:100%;margin:8px 0;font-size:9.8pt;break-inside:avoid}
th{background:#1B2333;color:#fff;text-align:left;padding:5px 8px;font-weight:600}
td{padding:5px 8px;border-bottom:1px solid #dde3ec;vertical-align:top}
code{font-family:'JetBrains Mono',monospace;font-size:9pt;background:#EAF0FB;padding:0 3px;border-radius:3px}
"""
open(dst, 'w', encoding='utf-8').write(
    f'<!doctype html><html><head><meta charset="utf-8"><title>Quick Summary</title>'
    f'<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono&display=block" rel="stylesheet">'
    f'<style>{css}</style></head><body>' + '\n'.join(out) + '</body></html>')
