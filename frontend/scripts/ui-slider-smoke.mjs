/**
 * Per-line editing: whole-line shift and the two-handle character slider.
 *
 * Runs against a real project, so the timeline is restored at the end — the
 * point is to prove the controls move the numbers, not to leave a changed
 * timeline behind.
 */
import puppeteer from "puppeteer-core";

const BASE = process.env.BASE ?? "http://127.0.0.1:8091";
const PROJECT = process.env.PROJECT ?? "prj_0caf07bdf59a46f48c6998ac55a294c4";

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

const readTimeline = async (page) =>
  page.evaluate(async (id) => {
    const response = await fetch(`/api/projects/${id}`);
    return (await response.json()).timeline;
  }, PROJECT);

/** Put the timeline back exactly as it was found (edits here are real saves). */
const restoreTimeline = async (page, timeline) =>
  page.evaluate(
    async (id, value) => {
      const response = await fetch(`/api/projects/${id}/timeline`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ timeline: value }),
      });
      if (!response.ok) throw new Error(`restore failed: ${response.status}`);
    },
    PROJECT,
    timeline,
  );

try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 1100 });
  const errors = [];
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  page.on("pageerror", (e) => errors.push(String(e)));

  await page.goto(`${BASE}/projects/${PROJECT}`, { waitUntil: "networkidle0" });
  await sleep(2500);

  const before = await readTimeline(page);
  const index = before.lines.findIndex((line) => line.words.length >= 3);
  if (index < 0) throw new Error("no line with characters to test");
  const original = before.lines[index];

  // --- selecting a row shows the slider ------------------------------------
  await page.evaluate((i) => document.querySelectorAll(".line-row")[i].click(), index);
  await sleep(300);
  const slider = await page.$(".line-row.selected .word-slider");
  check("the selected line shows a two-handle slider", slider !== null);
  const handles = await page.$$eval(".line-row.selected .word-slider-handle", (els) => els.length);
  check("the slider has a left and a right handle", handles === 2, `${handles} handles`);

  // --- whole-line shift ----------------------------------------------------
  await page.evaluate(() => {
    const button = [...document.querySelectorAll(".line-row.selected .toolbar button")].find(
      (el) => el.textContent.trim() === "+0.1s",
    );
    button.click();
  });
  await sleep(1800);
  const shifted = await readTimeline(page);
  const line = shifted.lines[index];
  const delta = line.start - original.start;
  check("整行 +0.1s moves the line start", Math.abs(delta - 0.1) < 0.005, `${original.start} → ${line.start}`);
  const wordsMoved = original.words.every((word, i) => Math.abs(line.words[i].start - (word.start + delta)) < 0.005);
  check("整行 +0.1s moves every character by the same amount", wordsMoved);
  const gapBefore = original.words[1].start - original.words[0].start;
  const gapAfter = line.words[1].start - line.words[0].start;
  check("整行平移 keeps the spacing between characters", Math.abs(gapBefore - gapAfter) < 0.005);

  await page.evaluate(() => {
    const button = [...document.querySelectorAll(".line-row.selected .toolbar button")].find(
      (el) => el.textContent.trim() === "-0.1s",
    );
    button.click();
  });
  await sleep(1800);
  const back = await readTimeline(page);
  check(
    "整行 -0.1s puts the line back",
    Math.abs(back.lines[index].start - original.start) < 0.005,
    `${back.lines[index].start}`,
  );

  // --- multi-select --------------------------------------------------------
  const chips = await page.$$(".line-row.selected .word-chip");
  await chips[1].click();
  await page.keyboard.down("Shift");
  await chips[2].click();
  await page.keyboard.up("Shift");
  await sleep(200);
  const selectedChips = await page.$$eval(".line-row.selected .word-chip.selected", (els) => els.length);
  check("Shift+点击 selects a run of characters", selectedChips >= 2, `${selectedChips} selected`);
  const label = await page.$eval(".line-row.selected .word-slider-label", (el) => el.textContent);
  check("the slider says how many characters it will move", /已选 \d+ 字/.test(label ?? ""), label ?? "");

  // --- dragging the right handle lengthens the run -------------------------
  const beforeDrag = (await readTimeline(page)).lines[index];
  const box = await page.$eval(".line-row.selected .word-slider-bar", (el) => {
    const rect = el.getBoundingClientRect();
    return { x: rect.x, y: rect.y, width: rect.width, height: rect.height };
  });
  const handle = await page.$eval(".line-row.selected .word-slider-handle.end", (el) => {
    const rect = el.getBoundingClientRect();
    return { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 };
  });
  await page.mouse.move(handle.x, handle.y);
  await page.mouse.down();
  await page.mouse.move(handle.x + box.width * 0.06, handle.y, { steps: 8 });
  await page.mouse.up();
  await sleep(1800);
  const dragged = await readTimeline(page);
  const grown = dragged.lines[index].words[2].end - beforeDrag.words[2].end;
  check("dragging the right handle lengthens the selected run", grown > 0.05, `+${grown.toFixed(2)}s`);
  const untouched = Math.abs(dragged.lines[index].words[0].start - beforeDrag.words[0].start) < 0.005;
  check("characters outside the selection keep their timing", untouched);
  const marked = dragged.lines[index].words.slice(1, 3).every((word) => word.source === "manual");
  check("dragged characters are marked as manual, not ASR", marked);

  // --- undo steps back over the whole drag --------------------------------
  await page.click("body");
  await page.keyboard.down("Control");
  await page.keyboard.press("z");
  await page.keyboard.up("Control");
  await sleep(1800);
  const undone = await readTimeline(page);
  check(
    "undo restores the drag in one step",
    Math.abs(undone.lines[index].words[2].end - beforeDrag.words[2].end) < 0.005,
    `${undone.lines[index].words[2].end} vs ${beforeDrag.words[2].end}`,
  );
  check(
    "undo leaves the line where it started",
    Math.abs(undone.lines[index].start - original.start) < 0.005,
    `${undone.lines[index].start}`,
  );

  // --- dragging the fill moves the whole run, shape intact -----------------
  const beforeMove = (await readTimeline(page)).lines[index];
  const fill = await page.$eval(".line-row.selected .word-slider-fill", (el) => {
    const rect = el.getBoundingClientRect();
    return { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 };
  });
  await page.mouse.move(fill.x, fill.y);
  await page.mouse.down();
  await page.mouse.move(fill.x + box.width * 0.04, fill.y, { steps: 8 });
  await page.mouse.up();
  await sleep(1800);
  const moved = await readTimeline(page);
  const shiftA = moved.lines[index].words[1].start - beforeMove.words[1].start;
  const shiftB = moved.lines[index].words[2].start - beforeMove.words[2].start;
  check("dragging the fill moves the whole run", shiftA > 0.02, `+${shiftA.toFixed(3)}s`);
  check("the run keeps its internal spacing", Math.abs(shiftA - shiftB) < 0.005, `${shiftA} vs ${shiftB}`);
  check(
    "the character before the run stays put",
    Math.abs(moved.lines[index].words[0].start - beforeMove.words[0].start) < 0.005,
  );

  await page.click("body");
  await page.keyboard.down("Control");
  await page.keyboard.press("z");
  await page.keyboard.up("Control");
  await sleep(1800);
  const backAgain = await readTimeline(page);
  check(
    "undo puts the run back after a move drag",
    Math.abs(backAgain.lines[index].words[1].start - original.words[1].start) < 0.005,
    `${backAgain.lines[index].words[1].start}`,
  );

  // --- 选中均分: even split inside the run's own window --------------------
  const beforeEven = (await readTimeline(page)).lines[index];
  await page.evaluate(() => {
    const button = [...document.querySelectorAll(".line-row.selected .word-slider-actions button")].find(
      (el) => el.textContent.trim() === "选中均分",
    );
    button.click();
  });
  await sleep(1800);
  const even = (await readTimeline(page)).lines[index];
  const windowStart = beforeEven.words[1].start;
  const windowEnd = beforeEven.words[2].end;
  const run = even.words.slice(1, 3);
  check(
    "选中均分 keeps the run inside its old window",
    Math.abs(run[0].start - windowStart) < 0.02 && Math.abs(run[1].end - windowEnd) < 0.02,
    `${run[0].start}..${run[1].end} vs ${windowStart}..${windowEnd}`,
  );
  const each = (windowEnd - windowStart) / 2;
  check(
    "选中均分 gives every selected character the same share",
    run.every((word, i) => Math.abs(word.start - (windowStart + each * i)) < 0.02),
    run.map((word) => word.start).join(", "),
  );
  check("选中均分 marks the characters as manual", run.every((word) => word.source === "manual"));

  await page.click("body");
  await page.keyboard.down("Control");
  await page.keyboard.press("z");
  await page.keyboard.up("Control");
  await sleep(1800);
  const afterUndoEven = await readTimeline(page);
  check(
    "undo puts the run back after 选中均分",
    Math.abs(afterUndoEven.lines[index].words[1].start - beforeEven.words[1].start) < 0.005,
  );

  // --- 选中平移 buttons ----------------------------------------------------
  const beforeNudge = (await readTimeline(page)).lines[index];
  await page.evaluate(() => {
    const button = [...document.querySelectorAll(".line-row.selected .word-slider-actions button")].find(
      (el) => el.textContent.trim() === "选中 +0.05s",
    );
    button.click();
  });
  await sleep(1800);
  const nudged = (await readTimeline(page)).lines[index];
  const nudgeA = nudged.words[1].start - beforeNudge.words[1].start;
  const nudgeB = nudged.words[2].start - beforeNudge.words[2].start;
  check("选中 +0.05s moves the whole selection", Math.abs(nudgeA - 0.05) < 0.005, `${nudgeA}`);
  check("选中 +0.05s keeps the selection's spacing", Math.abs(nudgeA - nudgeB) < 0.005);
  check(
    "选中 +0.05s leaves the character before it alone",
    Math.abs(nudged.words[0].start - beforeNudge.words[0].start) < 0.005,
  );
  await page.click("body");
  await page.keyboard.down("Control");
  await page.keyboard.press("z");
  await page.keyboard.up("Control");
  await sleep(1800);
  const afterUndoNudge = await readTimeline(page);
  check(
    "undo puts the selection back after 选中平移",
    Math.abs(afterUndoNudge.lines[index].words[1].start - beforeNudge.words[1].start) < 0.005,
  );

  // --- 跟随播放 toggle -----------------------------------------------------
  const followButton = async () =>
    page.evaluate(() => {
      const button = [...document.querySelectorAll(".toolbar button")].find((el) =>
        (el.textContent ?? "").includes("跟随播放"),
      );
      return button ? { pressed: button.getAttribute("aria-pressed"), text: button.textContent.trim() } : null;
    });
  const followOn = await followButton();
  check("the editor follows the playing line by default", followOn?.pressed === "true", JSON.stringify(followOn));
  await page.evaluate(() => {
    const button = [...document.querySelectorAll(".toolbar button")].find((el) =>
      (el.textContent ?? "").includes("跟随播放"),
    );
    button.click();
  });
  await sleep(200);
  const followOff = await followButton();
  check("跟随播放 can be switched off", followOff?.pressed === "false", JSON.stringify(followOff));
  await page.evaluate(() => {
    const button = [...document.querySelectorAll(".toolbar button")].find((el) =>
      (el.textContent ?? "").includes("跟随播放"),
    );
    button.click();
  });
  await sleep(200);
  check("跟随播放 switches back on", (await followButton())?.pressed === "true");

  await page.screenshot({ path: "/tmp/shot-ui-slider.png" });
  check("no console errors", errors.length === 0, JSON.stringify(errors));

  // The page autosaves one second after the last edit, so a restore sent
  // before that is overwritten by it. Let the pending save land first, then
  // reload so the tab has nothing left to write.
  await sleep(2000);
  await restoreTimeline(page, before);
  await page.goto(`${BASE}/projects/${PROJECT}`, { waitUntil: "networkidle0" });
  await sleep(1500);
  const restoredTimeline = await readTimeline(page);
  check("the script puts the timeline back", JSON.stringify(restoredTimeline.lines) === JSON.stringify(before.lines));
} finally {
  await browser.close();
}

console.log(failures.length ? `\n${failures.length} check(s) failed` : "\nall checks passed");
process.exit(failures.length ? 1 : 0);
