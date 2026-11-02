/**
 * Read-only smoke test for the reference timeline and the 逐字打轴 toolbar.
 *
 * It never edits lyrics or timings, so it is safe to point at a real project:
 *   PROJECT=prj_xxx node scripts/ui-reference-smoke.mjs
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
  args: ["--no-sandbox", "--disable-gpu", "--autoplay-policy=no-user-gesture-required"],
});

try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 1100 });
  const errors = [];
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  page.on("pageerror", (e) => errors.push(String(e)));

  await page.goto(`${BASE}/projects/${PROJECT}`, { waitUntil: "networkidle0" });
  await sleep(2500);
  const project = await page.evaluate(async (id) => (await fetch(`/api/projects/${id}`)).json(), PROJECT);

  // --- the source timeline is shown next to the lyric ----------------------
  const hints = await page.$$eval(".line-row .reference-time", (els) => els.map((el) => el.textContent));
  const expected = (project.reference_lines ?? []).length;
  check(
    "each line with a source timestamp shows a 参考 badge",
    expected > 0 ? hints.length >= expected : hints.length === 0,
    `${hints.length} badges for ${expected} reference lines`,
  );

  if (hints.length) {
    // Clicking the hint seeks the audio to the source's own time.
    const first = await page.$eval(".line-row .reference-time", (el) => el.textContent ?? "");
    const readoutBefore = await page.$eval(".time-readout", (el) => el.textContent);
    await page.click(".line-row .reference-time");
    await sleep(600);
    const readoutAfter = await page.$eval(".time-readout", (el) => el.textContent);
    check("clicking a 参考 badge seeks the audio", readoutBefore !== readoutAfter, `${first} → ${readoutAfter}`);
  }

  // --- original lyrics download -------------------------------------------
  const download = await page.$eval(".card:last-child a[download]", (el) => ({
    href: el.getAttribute("href"),
    text: el.textContent,
  }));
  check("原始歌词下载按钮存在", /lyrics\/export\?format=(lrc|txt)/.test(download.href ?? ""), download.text);
  check(
    "带时间轴的歌词源下载 LRC",
    project.lyrics_reference ? download.href.endsWith("format=lrc") : download.href.endsWith("format=txt"),
    download.href,
  );

  // --- 逐字打轴 controls ----------------------------------------------------
  const toolbar = await page.$$eval(".toolbar button", (els) => els.map((el) => el.textContent ?? ""));
  check("工具栏有「全部拆成逐字」", toolbar.some((text) => text.includes("全部拆成逐字")));
  check("工具栏有「逐字打轴（按行均分）」", toolbar.some((text) => text.includes("逐字打轴")));
  const runButton = await page.$eval(".card:last-child button.primary", (el) => el.textContent ?? "");
  check("对齐按钮写明逐字打轴", runButton.includes("逐字打轴"), runButton);

  await page.screenshot({ path: "/tmp/shot-reference.png" });
  check("no console errors", errors.length === 0, JSON.stringify(errors));
} finally {
  await browser.close();
}

console.log(failures.length ? `\n${failures.length} check(s) failed` : "\nall checks passed");
process.exit(failures.length ? 1 : 0);
