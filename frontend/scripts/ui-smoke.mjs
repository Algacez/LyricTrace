import puppeteer from "puppeteer-core";

// End-to-end check of the deployed WebUI: list, editor, playback + karaoke
// highlight, an edit that round-trips through the autosave API, and export.
// BASE=http://127.0.0.1:8091 PROJECT=prj_... node scripts/ui-smoke.mjs
const BASE = process.env.BASE ?? "http://localhost:5174";
const PROJECT = process.env.PROJECT ?? "";

const failures = [];
function check(name, ok, detail = "") {
  console.log(`${ok ? "PASS" : "FAIL"} ${name}${detail ? ` — ${detail}` : ""}`);
  if (!ok) failures.push(name);
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const browser = await puppeteer.launch({
  executablePath: "/usr/bin/google-chrome",
  headless: true,
  args: ["--no-sandbox", "--disable-gpu", "--autoplay-policy=no-user-gesture-required"],
});

try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 1100 });
  const errors = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(msg.text());
  });
  page.on("pageerror", (err) => errors.push(String(err)));

  // --- project list -------------------------------------------------------
  await page.goto(`${BASE}/`, { waitUntil: "networkidle0" });
  await page.screenshot({ path: "/tmp/ui-list.png" });
  const cards = await page.$$eval(".project-card", (els) => els.length);
  check("list renders project cards", cards > 0, `${cards} cards`);

  // --- editor -------------------------------------------------------------
  await page.goto(`${BASE}/projects/${PROJECT}`, { waitUntil: "networkidle0" });
  await sleep(2500);
  await page.screenshot({ path: "/tmp/ui-editor.png" });
  const rows = await page.$$eval(".line-row", (els) => els.length);
  check("editor renders every line", rows > 0, `${rows} rows`);

  const duration = await page.$eval(".time-readout", (el) => el.textContent);
  check("audio loads into the waveform", /\/\s*\d+:\d\d/.test(duration) && !/\/ 00:00$/.test(duration), duration);

  // --- playback + karaoke highlight --------------------------------------
  // Select the first line, which also seeks to its start; playing from there
  // puts the playhead inside a line so the karaoke strip has something to sing.
  await page.click(".line-row");
  await sleep(300);
  const lineStart = await page.$eval(".line-row.selected .time-input", (el) => el.value);
  check("selecting a line seeks to it", !/^00:00\.000$/.test(lineStart), lineStart);

  await page.click(".transport button.primary");
  await sleep(1600);
  const playing = await page.$eval(".transport button.primary", (el) => el.textContent);
  const time = await page.$eval(".time-readout", (el) => el.textContent);
  const karaoke = await page.evaluate(() => {
    const line = document.querySelector(".karaoke-line");
    if (!line) return null;
    return {
      text: line.textContent ?? "",
      sung: line.querySelectorAll(".karaoke-word.sung").length,
      words: line.querySelectorAll(".karaoke-word").length,
    };
  });
  await page.screenshot({ path: "/tmp/ui-playing.png" });
  check("playback starts", playing.includes("暂停"), playing);
  check("playhead advances", !/^00:00\.\d+ \/ /.test(time), time);
  check("karaoke line highlights sung characters", karaoke !== null && karaoke.sung > 0, JSON.stringify(karaoke));

  await page.click(".transport button.primary");
  await sleep(300);
  const paused = await page.$eval(".transport button.primary", (el) => el.textContent);
  check("playback pauses", paused.includes("播放"), paused);

  // --- edit + autosave round-trip ----------------------------------------
  const before = await (await fetch(`${BASE}/api/projects/${PROJECT}`)).json();
  const originalStart = before.timeline.lines[0].start;
  const buttons = await page.$$eval(".line-row.selected button", (els) => els.map((e) => e.textContent));
  check("selected row exposes the edit tools", buttons.includes("按歌词拆分逐字"), JSON.stringify(buttons));

  // Shift the first line by +0.5 s through the start field.
  const startField = await page.$(".line-row.selected .time-input");
  const shifted = originalStart === null ? 1 : Math.round((originalStart + 0.5) * 100) / 100;
  await typeTime(page, startField, formatMmss(shifted));
  await sleep(2500);
  const afterEdit = await (await fetch(`${BASE}/api/projects/${PROJECT}`)).json();
  check(
    "edit is autosaved",
    Math.abs(afterEdit.timeline.lines[0].start - shifted) < 0.011,
    `${afterEdit.timeline.lines[0].start} vs ${shifted}`,
  );

  // Put it back so the smoke test leaves the project as it found it.
  await typeTime(page, startField, formatMmss(originalStart));
  await sleep(2500);
  const restored = await (await fetch(`${BASE}/api/projects/${PROJECT}`)).json();
  check(
    "restoring the value round-trips",
    originalStart === null
      ? restored.timeline.lines[0].start === null
      : Math.abs(restored.timeline.lines[0].start - originalStart) < 0.011,
    `${restored.timeline.lines[0].start} vs ${originalStart}`,
  );

  // --- export -------------------------------------------------------------
  for (const format of ["json", "lrc", "ass"]) {
    const response = await fetch(`${BASE}/api/projects/${PROJECT}/export?format=${format}`);
    const body = await response.text();
    check(
      `export ${format.toUpperCase()} downloads`,
      response.ok && body.trim().length > 0,
      `${response.status} ${body.length} bytes`,
    );
  }
  const exported = await (await fetch(`${BASE}/api/projects/${PROJECT}/export?format=json`)).json();
  const timed = exported.lines.filter((line) => line.start !== null).length;
  check("export JSON carries timed lines", timed > 0, `${timed} timed`);

  check("no console errors", errors.length === 0, JSON.stringify(errors));
} finally {
  await browser.close();
}

if (failures.length) {
  console.log(`\n${failures.length} check(s) failed: ${failures.join(", ")}`);
  process.exitCode = 1;
} else {
  console.log("\nall checks passed");
}

function formatMmss(seconds) {
  if (seconds === null || seconds === undefined) return "";
  const minutes = Math.floor(seconds / 60);
  return `${String(minutes).padStart(2, "0")}:${(seconds - minutes * 60).toFixed(2).padStart(5, "0")}`;
}

/** Replace a TimeField's contents and commit it (Enter blurs, which parses). */
async function typeTime(page, handle, text) {
  await handle.evaluate((el) => el.select());
  await page.keyboard.type(text);
  await page.keyboard.press("Enter");
  await sleep(300);
}
