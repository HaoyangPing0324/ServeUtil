"""Render the bilingual documentation from the repository's Markdown sources."""

import argparse
import json
import shutil
from html import escape
from pathlib import Path
from string import Template
from urllib.parse import urlsplit

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
THEME = ROOT / "packaging/portal"
# The same route in each language always refers to the same guide.
PAGES = {
    "index": ("index", "Overview", "概览"),
    "start": ("start", "First capture", "首次采集"),
    "downloads": ("downloads", "Installation", "安装"),
    "torch-profiler": ("torch-profiler", "PyTorch integration", "PyTorch 集成"),
    "capture": ("capture", "vLLM & SGLang", "vLLM 与 SGLang"),
    "metrics": ("metrics", "Metrics & calibration", "指标与标定"),
    "tools": ("tools", "Integrations", "工具集成"),
    "validation": ("validation", "Model validation", "模型验证"),
}
GROUPS = [
    ("Getting started", "入门", ("index", "downloads", "start")),
    ("Capture", "采集", ("torch-profiler", "capture", "metrics", "tools", "validation")),
]
def language_pages(site, zh, revision):
    def tr(en, cn):
        return cn if zh else en

    root = "../" if zh else ""
    out = site / "zh" if zh else site
    out.mkdir(exist_ok=True, parents=True)
    source = ROOT / ("docs" if zh else "docs/en")
    viewer = "https://haoyangping0324.github.io/ServeUtil-Viewer/viewer.html" + ("?lang=zh" if zh else "")
    docs = { (source / (page[0] + ".md")).resolve(): slug + ".html" for slug, page in PAGES.items() }
    template = Template((THEME / "layout.html").read_text(encoding="utf-8"))

    def write(slug, title, body):
        contents = template.substitute(
            lang=tr("en", "zh-CN"), title=escape(title), root=root, body=body,
            otherlang=tr("zh-CN", "en"), alternate=tr("zh/", "../") + slug + ".html",
            language=tr("简体中文", "English"), viewer=viewer, skip=tr("Skip to content", "跳到正文"),
            primary=tr("Main navigation", "主导航"), guide=tr("Documentation", "文档"),
            downloads=tr("Downloads", "下载"), launch=tr("Open viewer", "打开查看器"),
        )
        (out / (slug + ".html")).write_text(contents, encoding="utf-8")

    def nav(slug):
        sections = []
        for en, cn, names in GROUPS:
            sections.append(f"<b>{tr(en, cn)}</b>")
            for name in names:
                current = ' aria-current="page"' if name == slug else ""
                sections.append(f'<a href="{name}.html"{current}>{tr(*PAGES[name][1:])}</a>')
        return "".join(sections)

    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    for slug, (file, en, cn) in PAGES.items():
        path = source / (file + ".md")
        tokens = md.parse(path.read_text(encoding="utf-8"))
        toc = []
        for i, token in enumerate(tokens):
            if token.type == "heading_open" and token.tag == "h2":
                anchor = "section-" + str(len(toc) + 1)
                token.attrSet("id", anchor)
                toc.append(f'<a href="#{anchor}">{escape(tokens[i + 1].content)}</a>')
            for child in token.children or []:
                if child.type != "link_open":
                    continue
                href = child.attrGet("href")
                url = urlsplit(href)
                if url.scheme or not url.path:
                    continue
                target = (path.parent / url.path).resolve()
                if target in docs:
                    link = docs[target]
                elif target.parent == ROOT / "examples" and target.is_file():
                    if target.name.endswith(".report.json"):
                        link = root + target.name.replace(".report.json", ".html")
                    elif target.suffix == ".html":
                        link = root + target.name
                    else:
                        link = root + "examples/" + target.name
                    if zh and link.endswith(".html"):
                        link += "?lang=zh"
                else:
                    raise ValueError(f"Unknown documentation link {href} in {path}")
                child.attrSet("href", link + ("#" + url.fragment if url.fragment else ""))
        article = md.renderer.render(tokens, md.options, {}).replace("<table>", '<div class="tablewrap"><table>').replace("</table>", "</table></div>")
        menu = nav(slug)
        write(slug, tr(en, cn), f'<main class="frame" id="content"><nav class="side">{menu}</nav><details class="mobile-nav"><summary>{tr("Browse guides","浏览指南")}</summary><nav>{menu}</nav></details><article class="article">{article}</article><nav class="toc"><b>{tr("On this page","本页内容")}</b>{"".join(toc)}</nav></main>')



# Preserve bookmarks made before the viewer moved to its own Pages site.
def viewer_redirects(site):
    base = "https://haoyangping0324.github.io/ServeUtil-Viewer/"
    timelines = ("viewer", "inference-example", "native-example", "h20-decode",
                 "h20-allreduce", "sglang-erdma", "sglang-pd")
    for lang in ("", "zh/"):
        for page in (*timelines, "viewer-guide", "decode", "bandwidth", "pd"):
            target = base + (lang if page not in timelines else "") + page + ".html"
            fallback = target + ("?lang=zh" if lang and page in timelines else "")
            script = (
                f"const url = new URL({json.dumps(target)});url.search=location.search;"
                + ("if(!url.searchParams.has('lang'))url.searchParams.set('lang','zh');"
                   if lang and page in timelines else "")
                + "url.hash=location.hash;location.replace(url.href);"
            )
            (site / lang / (page + ".html")).write_text(
                '<!doctype html><meta charset="utf-8"><title>ServeUtil Viewer</title>'
                f'<script>{script}</script><a href="{escape(fallback)}">ServeUtil Viewer</a>',
                encoding="utf-8",
            )


def build_docs(site, revision="local"):
    assets = site / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name in ("wordmark.png", "icon.png"):
        shutil.copyfile(ROOT / "assets" / name, assets / name)
    shutil.copyfile(THEME / "site.css", assets / "docs.css")
    for asset in ("github.svg", "octicons-LICENSE.txt", "docs.js"):
        shutil.copyfile(THEME / asset, assets / asset)
    shutil.copyfile(ROOT / "LICENSE", site / "LICENSE.txt")
    examples = site / "examples"
    examples.mkdir(exist_ok=True)
    for path in (ROOT / "examples").glob("*.json"):
        shutil.copyfile(path, examples / path.name)
    for zh in (False, True):
        language_pages(site, zh, revision)
    viewer_redirects(site)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("site", type=Path)
    args = parser.parse_args()
    build_docs(args.site)
