const zh = document.documentElement.lang === "zh-CN";
for (const block of document.querySelectorAll("pre > code")) {
  if (!navigator.clipboard?.writeText) break;
  const box = document.createElement("div");
  box.className = "code-block";
  const pre = block.parentElement;
  pre.before(box);
  box.append(pre);
  const copy = document.createElement("button");
  copy.textContent = zh ? "复制" : "Copy";
  copy.setAttribute("aria-label", zh ? "复制代码" : "Copy code");
  copy.onclick = async () => {
    try {
      await navigator.clipboard.writeText(block.textContent);
      copy.textContent = zh ? "已复制" : "Copied";
    } catch {
      const range = document.createRange();
      range.selectNodeContents(block);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      copy.textContent = zh ? "已选中" : "Selected";
    }
    setTimeout(() => { copy.textContent = zh ? "复制" : "Copy"; }, 1600);
  };
  box.append(copy);
}
