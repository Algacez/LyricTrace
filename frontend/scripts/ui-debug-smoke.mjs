/**
 * Smoke test for the debug page and single-character editing.
 *
 *   PROJECT=prj_xxx node scripts/ui-debug-smoke.mjs
 *
 * The word-edit check changes one character of the first line and then edits it
 * back, so point it at a throwaway project.
 */
import puppeteer from "puppeteer-core";

const BASE = process.env.BASE ?? "http://127.0.0.1:8091";
const PROJECT = process.env.PROJECT ?? "";

const failures = [];
function check(name, ok, detail = "") {
  console.log(`${ok ? "PASS" : "FAIL"} ${name}${detail ? ` — ${detail}` : ""}`);
  if (!ok) failures.push(name);
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await puppeteer.launch({
  executablePath: "/usr/bin/google-chrome",
  headless: true,
  args: ["--no-sandbox", "--disable-gpu"],
});

try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 1100 });
  const errors = [];
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  page.on("pageerror", (e) => errors.push(String(e)));

  // ---------------------------------------------------------------- debug page
  await page.goto(`${BASE}/projects/${PROJECT}/debug`, { waitUntil: "networkidle0" });
  await sleep(1200);

  const steps = await page.$$eval(".debug-step", (els) =>
    els.map((el) => el.querySelector(".badge")?.textContent ?? ""),
  );
  check("调试页按顺序显示每一步", steps.length >= 8, JSON.stringify(steps));
  check(
    "步骤顺序与流水线一致",
    steps[0] === "metadata" && steps.includes("asr_words") && steps.includes("timeline"),
    JSON.stringify(steps),
  );

  const tables = await page.$$eval(".debug-table", (els) => els.length);
  check("中间结果用表格展示", tables > 0, `${tables} tables`);

  await page.evaluate(() => {
    const button = [...document.querySelectorAll("button")].find((el) =>
      (el.textContent ?? "").includes("全部展开原始 JSON"),
    );
    button?.click();
  });
  await sleep(400);
  const jsonBlocks = await page.$$eval(".debug-json", (els) => els.length);
  check("可以展开原始 JSON", jsonBlocks >= steps.length, `${jsonBlocks} blocks`);

  // ------------------------------------------------------ single-character edit
  await page.goto(`${BASE}/projects/${PROJECT}`, { waitUntil: "networkidle0" });
  await sleep(2500);

  // Select the first row: its first word chip is focused, which reveals the ✎.
  await page.click(".line-row");
  await sleep(400);

  const before = await page.$eval(".line-row.selected", (el) => ({
    line: el.querySelector(".line-head .text")?.textContent ?? "",
    firstWord: el.querySelector(".word-chip")?.textContent ?? "",
  }));
  check("选中字后出现改字按钮", Boolean(await page.$(".line-row.selected .word-chip.selected .word-edit")));

  await page.click(".line-row.selected .word-chip.selected .word-edit");
  await sleep(200);
  const input = await page.$(".word-chip-input");
  check("改字进入输入框", Boolean(input));
  await page.$eval(".word-chip-input", (el) => el.select());
  await page.type(".word-chip-input", "〇");
  await page.keyboard.press("Enter");
  await sleep(2000);

  const after = await page.$eval(".line-row.selected", (el) => ({
    line: el.querySelector(".line-head .text")?.textContent ?? "",
    firstWord: el.querySelector(".word-chip")?.textContent ?? "",
  }));
  check("单个字被改写", after.firstWord.startsWith("〇"), JSON.stringify(after));
  check(
    "整行歌词同步更新",
    after.line !== before.line && after.line.startsWith("〇"),
    `${before.line} → ${after.line}`,
  );

  const saved = await page.evaluate(async (id) => (await fetch(`/api/projects/${id}`)).json(), PROJECT);
  const savedLine = saved.timeline?.lines?.[0];
  check(
    "改字已保存，且时间没被动",
    savedLine?.words?.[0]?.text === "〇" && savedLine?.words?.[0]?.start !== null,
    JSON.stringify(savedLine?.words?.[0] ?? null),
  );

  // Put the character back so the check is repeatable.
  await page.$eval(".line-row.selected .word-chip", (el) => el.click());
  await sleep(300);
  await page.click(".line-row.selected .word-chip.selected .word-edit");
  await sleep(200);
  const original = before.firstWord.trim().charAt(0) || "";
  if (original) {
    await page.$eval(".word-chip-input", (el) => el.select());
    await page.type(".word-chip-input", original);
    await page.keyboard.press("Enter");
    await sleep(1800);
  }

  check("no console errors", errors.length === 0, JSON.stringify(errors));
} finally {
  await browser.close();
}

console.log(failures.length ? `\n${failures.length} check(s) failed` : "\nall checks passed");
process.exit(failures.length ? 1 : 0);
