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

  // --- 下一处待确认 must select AND scroll the row into view ---------------
  const before = await page.evaluate(() => {
    const list = document.querySelector(".timeline-editor");
    return { scroll: list.scrollTop, selected: document.querySelector(".line-row.selected")?.textContent ?? null };
  });
  await page.click(".flag-action");
  // scrollIntoView is smooth; a taller row takes longer to settle.
  await sleep(1500);
  const after = await page.evaluate(() => {
    const list = document.querySelector(".timeline-editor");
    const row = document.querySelector(".line-row.selected");
    const listRect = list.getBoundingClientRect();
    const rowRect = row?.getBoundingClientRect();
    return {
      scroll: list.scrollTop,
      selected: row?.textContent ?? null,
      visible: rowRect ? rowRect.top >= listRect.top - 2 && rowRect.bottom <= listRect.bottom + 2 : false,
      flagged: row?.classList.contains("flagged") ?? false,
    };
  });
  check("下一处待确认 selects a row", after.selected !== null && after.selected !== before.selected, (after.selected ?? "").slice(0, 30));
  check("下一处待确认 scrolls that row into view", after.visible, `scrollTop ${before.scroll} → ${after.scroll}`);
  check("the jumped-to row is a flagged one", after.flagged);

  // --- 上一处待确认 --------------------------------------------------------
  await page.click(".flag-action");
  await sleep(500);
  const prev = await page.$eval(".line-row.selected", (el) => el.textContent);
  check("上一处待确认 goes back", prev !== after.selected);

  // --- inline text editing -------------------------------------------------
  await page.click(".line-row.selected .line-head .text");
  await sleep(200);
  const input = await page.$(".line-row.selected .line-text-input");
  check("clicking the selected line opens an inline editor", input !== null);

  const original = await page.$eval(".line-row.selected .line-text-input", (el) => el.value);
  // Triple-click does not reliably select an input's text under puppeteer; use the DOM API.
  const selectAll = () => page.$eval(".line-row.selected .line-text-input", (el) => el.select());
  await selectAll();
  await page.type(".line-row.selected .line-text-input", `${original}（测）`);
  await page.keyboard.press("Enter");
  await sleep(1800);
  const edited = await page.evaluate(() => {
    const row = document.querySelector(".line-row.selected");
    return {
      text: row?.querySelector(".text")?.textContent ?? "",
      words: row?.querySelectorAll(".word-chip").length ?? 0,
      flags: row?.className ?? "",
    };
  });
  check("editing the text updates the row", edited.text.includes("（测）"), edited.text.slice(0, 30));
  check("edited text is re-split into words", edited.words > 0, `${edited.words} words`);
  check("the stale 待确认 flag is cleared after editing", !edited.flags.includes("flagged"), edited.flags);

  // restore
  await page.click(".line-row.selected .line-head .text");
  await selectAll();
  await page.type(".line-row.selected .line-text-input", original);
  await page.keyboard.press("Enter");
  await sleep(1800);
  const restored = await page.$eval(".line-row.selected .text", (el) => el.textContent);
  check("restoring the text round-trips", restored === original, `${restored} vs ${original}`);

  // --- lyrics source chain in the panel ------------------------------------
  // Click by label: the panel's first button is the download link, not search.
  await page.evaluate(() => {
    const button = [...document.querySelectorAll("button")].find((el) =>
      (el.textContent ?? "").includes("查找歌词"),
    );
    if (!button) throw new Error("no 查找歌词 button");
    button.click();
  });
  await page.waitForFunction(() => document.querySelectorAll(".lyrics-candidate").length > 0, { timeout: 30000 });
  const badges = await page.$$eval(".lyrics-candidate .badge", (els) => els.map((el) => el.textContent));
  check("search results carry a source badge", badges.length > 0, JSON.stringify(badges.slice(0, 6)));

  await page.screenshot({ path: "/tmp/shot-ui2.png" });
  check("no console errors", errors.length === 0, JSON.stringify(errors));
} finally {
  await browser.close();
}

console.log(failures.length ? `\n${failures.length} check(s) failed` : "\nall checks passed");
process.exit(failures.length ? 1 : 0);
