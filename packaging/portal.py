"""Render the bilingual documentation from the repository's Markdown sources."""

import argparse
import os
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
    "start": ("start", "Quick start", "快速开始"),
    "downloads": ("downloads", "Install & download", "安装与下载"),
    "decode": ("recipes/decode", "Decode gaps", "Decode 空档"),
    "bandwidth": ("recipes/bandwidth", "Network utilization", "网络利用率"),
    "pd": ("recipes/pd", "KV handoffs", "KV 交接"),
    "torch-profiler": ("torch-profiler", "PyTorch integration", "PyTorch 集成"),
    "capture": ("capture", "vLLM & SGLang", "vLLM 与 SGLang"),
    "viewer-guide": ("viewer", "Using the viewer", "使用查看器"),
    "metrics": ("metrics", "Metrics & calibration", "指标与标定"),
    "tools": ("tools", "Integrations", "工具集成"),
    "validation": ("validation", "Model validation", "模型验证"),
}
GROUPS = [
    ("Getting started", "入门", ("index", "start", "downloads")),
    ("Hands-on recipes", "实测教程", ("decode", "bandwidth", "pd")),
    ("Reference", "使用指南", ("torch-profiler", "capture", "viewer-guide", "metrics", "tools", "validation")),
]
PLATFORMS = {
    "Windows x86-64": "ServeUtil-Windows-AMD64.zip",
    "macOS · Apple silicon": "ServeUtil-Darwin-arm64.tar.gz",
    "Linux x86-64": "ServeUtil-Linux-x86_64.tar.gz",
}


def language_pages(site, zh, revision):
    def tr(en, cn):
        return cn if zh else en

    root = "../" if zh else ""
    out = site / "zh" if zh else site
    out.mkdir(exist_ok=True, parents=True)
    source = ROOT / ("docs" if zh else "docs/en")
    viewer = root + "viewer.html" + ("?lang=zh" if zh else "")
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
        if slug == "downloads":
            links = []
            for platform, name in PLATFORMS.items():
                asset = site / "downloads" / name
                if asset.exists():
                    size = round(asset.stat().st_size / 1024**2, 1)
                    links.append(f'<div class="download"><a href="{root}downloads/{name}" download>{platform} ↓</a><span>{size} MiB · {escape(revision)}</span></div>')
            if not links:
                workflow = "https://github.com/HaoyangPing0324/ServeUtil/actions/workflows/desktop.yml"
                links.append(f'<p><a href="{workflow}">{tr("Download desktop builds from GitHub Actions (sign-in required)", "从 GitHub Actions 下载桌面构建（需登录）")} ↗</a></p>')
            article = article.replace('<h2 id="section-3">', "".join(links) + '<h2 id="section-3">')
        menu = nav(slug)
        write(slug, tr(en, cn), f'<main class="frame" id="content"><nav class="side">{menu}</nav><details class="mobile-nav"><summary>{tr("Browse guides","浏览指南")}</summary><nav>{menu}</nav></details><article class="article">{article}</article><nav class="toc"><b>{tr("On this page","本页内容")}</b>{"".join(toc)}</nav></main>')


def build_docs(site, revision="local"):
    assets = site / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name in ("wordmark.png", "icon.png", "h20-decode.png"):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("site", type=Path)
    parser.add_argument("--desktop-assets", required=True, type=Path)
    args = parser.parse_args()
    downloads = args.site / "downloads"
    downloads.mkdir(exist_ok=True)
    for name in PLATFORMS.values():
        matches = list(args.desktop_assets.rglob(name))
        if len(matches) != 1 or matches[0].stat().st_size == 0:
            raise ValueError(f"Expected one verified {name} archive")
        shutil.copyfile(matches[0], downloads / name)
    build_docs(args.site, os.getenv("GITHUB_SHA", "local")[:8])
