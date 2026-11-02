/**
 * Smoke test for the 逐字打轴 toolbar buttons. It edits timings, so point it at
 * a throwaway project:
 *   PROJECT=prj_xxx node scripts/ui-axis-smoke.mjs
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
  // The two bulk actions ask for confirmation before dropping existing timings.
  page.on("dialog", (dialog) => void dialog.accept());

  await page.goto(`${BASE}/projects/${PROJECT}`, { waitUntil: "networkidle0" });
  await sleep(2500);

  const clickToolbar = (label) =>
    page.evaluate((text) => {
      const button = [...document.querySelectorAll(".toolbar button")].find((el) =>
        (el.textContent ?? "").includes(text),
      );
      if (!button) throw new Error(`no toolbar button ${text}`);
      button.click();
    }, label);

  const snapshot = () =>
    page.evaluate(() => ({
      status: document.querySelector(".card h2 .badge")?.textContent ?? "",
      rows: [...document.querySelectorAll(".line-row")].map((row) => ({
        words: row.querySelectorAll(".word-chip").length,
        timed: row.querySelectorAll(".word-chip.timed").length,
      })),
    }));

  check("starting state has no per-word slots", (await snapshot()).rows.every((row) => row.words === 0));

  await clickToolbar("全部拆成逐字");
  await sleep(2000);
  const split = await snapshot();
  check(
    "全部拆成逐字 gives every line word slots",
    split.rows.length > 0 && split.rows.every((row) => row.words > 0),
    JSON.stringify(split.rows),
  );
  check("splitting does not invent word times", split.rows.every((row) => row.timed === 0), split.status);

  await clickToolbar("逐字打轴（按行均分）");
  await sleep(2000);
  const distributed = await snapshot();
  check(
    "逐字打轴 fills every word inside its line window",
    distributed.rows.every((row) => row.words > 0 && row.timed === row.words),
    JSON.stringify(distributed.rows),
  );
  check("the timeline becomes 逐字完整", distributed.status.includes("逐字完整"), distributed.status);

  const saved = await page.evaluate(async (id) => (await fetch(`/api/projects/${id}`)).json(), PROJECT);
  check(
    "the distributed words are saved as interpolated",
    saved.timeline.lines.every((line) => line.words.every((word) => word.source === "interpolated" && word.start !== null)),
  );

  // A line with a source timestamp offers 用参考时间, which is marked as a hint.
  await page.click(".line-row");
  await sleep(400);
  const applyButton = await page.$(".line-row.selected .toolbar button");
  const labels = await page.$$eval(".line-row.selected .toolbar button", (els) => els.map((el) => el.textContent));
  check("选中行可以套用参考时间", labels.some((text) => text?.includes("用参考时间")), JSON.stringify(labels));
  if (applyButton && labels.some((text) => text?.includes("用参考时间"))) {
    await page.evaluate(() => {
      const button = [...document.querySelectorAll(".line-row.selected .toolbar button")].find((el) =>
        (el.textContent ?? "").includes("用参考时间"),
      );
      button?.click();
    });
    await sleep(2000);
    const flagged = await page.$eval(".line-row.selected", (el) => ({
      badge: el.querySelector(".badge.flag")?.textContent ?? "",
      start: el.querySelector(".time-input")?.value ?? "",
    }));
    check("套用参考时间后标为参考定位", flagged.badge.includes("参考定位"), JSON.stringify(flagged));
    check("套用参考时间写入源时间", flagged.start.includes("00:01"), flagged.start);
  }

  check("no console errors", errors.length === 0, JSON.stringify(errors));
} finally {
  await browser.close();
}

console.log(failures.length ? `\n${failures.length} check(s) failed` : "\nall checks passed");
process.exit(failures.length ? 1 : 0);
