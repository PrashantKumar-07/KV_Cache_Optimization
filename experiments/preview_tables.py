#!/usr/bin/env python3
"""preview_tables.py — render every results/tables/*.tex table as one HTML page.

There is no LaTeX toolchain on this box, so .tex tables can't be compiled
to PDF here. This parses the tabular body + caption out of each .tex file
and writes results/tables/_preview.html — open it in any browser.

Usage (from the kv_cache directory):
    ../venv_kvcache/bin/python experiments/preview_tables.py
    # then open results/tables/_preview.html
"""

import glob
import html
import os
import re

TABDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "results", "tables")


def _balanced(text, start):
    """Return contents of the brace group opened at text[start] == '{'."""
    assert text[start] == "{"
    depth, out = 0, []
    for ch in text[start:]:
        if ch == "{":
            depth += 1
            if depth > 1:
                out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
            out.append(ch)
        else:
            out.append(ch)
    raise ValueError("unbalanced braces")


def tex2html_cell(cell):
    cell = cell.strip()
    # \textbf{X} -> <b>X</b> (innermost first)
    while True:
        m = re.search(r"\\textbf\{", cell)
        if not m:
            break
        inner = _balanced(cell, m.end() - 1)
        cell = (cell[:m.start()] + "<b>" + inner + "</b>"
                + cell[m.end() - 1 + len(inner) + 2:])
    cell = (cell.replace(r"$\times$", "×").replace(r"\times", "×")
            .replace(r"$\pm$", "±").replace(r"\pm", "±")
            .replace(r"\_", "_").replace(r"\%", "%")
            .replace("$", "").replace(r"\ ", " ").replace("~", " "))
    cell = re.sub(r"\\[a-zA-Z]+\*?", "", cell)  # drop stray commands
    cell = re.sub(r"[{}]", "", cell)
    return html.escape(cell, quote=False).replace("&lt;b&gt;", "<b>").replace(
        "&lt;/b&gt;", "</b>")


def parse_table(path):
    with open(path) as f:
        src = f.read()
    m = re.search(r"\\caption\{", src)
    caption = _balanced(src, m.end() - 1) if m else os.path.basename(path)
    caption = tex2html_cell(caption)
    body = src[src.index(r"\begin{tabular}"):src.index(r"\end{tabular}")]
    header_rows, body_rows, in_body = [], [], False
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith(r"\begin") or line.startswith(r"\toprule"):
            continue
        if line.startswith(r"\midrule"):
            in_body = True
            continue
        if line.startswith(r"\bottomrule") or line.startswith(r"\end"):
            break
        if line.startswith(r"\hline"):
            continue
        line = line.split(r"\\")[0]
        cells = [tex2html_cell(c) for c in line.split("&")]
        (body_rows if in_body else header_rows).append(cells)
    return caption, header_rows, body_rows


CSS = """
:root{color-scheme:light}
body{font-family:Georgia,serif;max-width:1000px;margin:2em auto;padding:0 1em;background:#ffffff;color:#222}
h1{font-size:1.4em;color:#111} h2{font-size:1.1em;margin-top:2.2em;color:#111}
table{border-collapse:collapse;margin:1em 0;font-size:.92em;background:#fff}
caption{text-align:left;font-style:italic;margin-bottom:.5em;color:#444}
thead td{border-top:2px solid #333;border-bottom:1px solid #333;font-weight:bold;padding:5px 10px;text-align:right;color:#111}
thead td:first-child,tbody td:first-child{text-align:left}
tbody td{padding:4px 10px;text-align:right;border-bottom:1px solid #eee;color:#222}
tbody tr:last-child td{border-bottom:2px solid #333}
a{color:#1a5fb4}
"""
META = "<meta charset='utf-8'><meta name='color-scheme' content='light'>"


def main():
    paths = sorted(glob.glob(os.path.join(TABDIR, "*.tex")))
    parts = ["<html><head>" + META + "<style>" + CSS + "</style></head><body>",
             "<h1>KV-Cache paper tables (rendered from results/tables/*.tex)</h1>"]
    for p in paths:
        try:
            caption, head, body = parse_table(p)
        except Exception as e:  # noqa: BLE001 — one bad table must not kill the page
            parts.append(f"<h2>{html.escape(os.path.basename(p))}</h2><p>parse failed: {e}</p>")
            continue
        parts.append(f"<h2>{html.escape(os.path.basename(p))}</h2>")
        parts.append("<table><caption>" + caption + "</caption><thead>")
        for r in head:
            parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
        parts.append("</thead><tbody>")
        for r in body:
            parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
        parts.append("</tbody></table>")
    parts.append("</body></html>")
    out = os.path.join(TABDIR, "_preview.html")
    with open(out, "w") as f:
        f.write("\n".join(parts))
    print(f"wrote {out} ({len(paths)} tables)")


if __name__ == "__main__":
    main()
