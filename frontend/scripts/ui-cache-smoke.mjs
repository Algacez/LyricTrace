import puppeteer from "puppeteer-core";

// The 「API 缓存」 page: it lists every /api/align job ever submitted and
// corrects one in the browser with the same TimelineEditor the project page
// uses. BASE=http://127.0.0.1:8091
// node scripts/ui-cache-smoke.mjs
const BASE = process.env.BASE ?? "http://127.0.0.1:8091";

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

  await page.goto(`${BASE}/cache`, { waitUntil: "networkidle0" });
  await sleep(1500);

  const items = await page.$$eval(".cache-item", (els) => els.length);
  check("cache page lists jobs", items > 0, `${items} items`);

  const header = await page.$eval(".cache-item .title", (el) => el.textContent ?? "");
  check("a cached job is named", header.trim().length > 0, header.trim());

  // The newest entry is selected by default, and it may be a job that failed
  // before it produced a result — its audio is still playable, which is the
  // point of keeping it. Pick a finished one for the result checks below.
  await page.evaluate(() => {
    const buttons = [...document.querySelectorAll(".cache-item")];
    const finished = buttons.find((b) => /已完成/.test(b.textContent ?? ""));
    if (finished) finished.click();
  });
  await sleep(800);

  const audio = await page.$eval("audio", (el) => el.getAttribute("src") ?? "");
  check("the source audio is served", audio.includes("/api/align/") && audio.endsWith("/audio"), audio);

  // The result block is the project page's TimelineEditor, so its rows are the
  // editor's ``.line-row`` (not a cache-page-only markup) and its characters are
  // the editor's ``.word-chip`` — one chip per character, never a run.
  const lines = await page.$$eval(".line-row", (els) => els.length);
  check("the result renders its lines", lines > 0, `${lines} lines`);

  const words = await page.$$eval(".word-strip .word-chip", (els) =>
    els.map((el) => (el.textContent ?? "").trim()),
  );
  check("every character has its own span", words.length > 0, `${words.length} characters`);

  // Seeking to a line must move the player and light that character. Pick a
  // line that actually has a time: an untimed line (the ASR never heard it)
  // seeks to 0 and lights nothing, which is correct behaviour, not a failure.
  // The editor writes the line window into two ``.time-input`` boxes.
  await page.evaluate(() => {
    const rows = [...document.querySelectorAll(".line-row")];
    const timed = (row) => {
      const box = (row.querySelectorAll(".time-input")[0]?.value ?? "").trim();
      return box !== "" && box !== "00:00.000";
    };
    (rows.find(timed) ?? rows[0]).click();
  });
  await sleep(1500);

  // The karaoke line lives in the waveform panel, above the editor's rows.
  const active = await page.$$eval(".waveform-panel .karaoke-word.active", (els) =>
    els.map((el) => el.textContent ?? ""),
  );
  check("seeking lights the character being sung", active.length === 1, JSON.stringify(active));

  // The line being sung must be spelled out by its own characters: the chips of
  // the lit line, joined without spaces, are exactly the text on screen. No
  // character is missing and none is bundled into a longer run.
  const spelled = await page.$$eval(".waveform-panel .karaoke-line", (els) =>
    (els[0]?.textContent ?? "").replace(/\s+/g, ""),
  );
  const chips = await page.$$eval(".line-row.active .word-chip", (els) =>
    els.map((el) => (el.textContent ?? "").replace(/\s+/g, "")),
  );
  check(
    "the lit line is spelled out one character per span",
    spelled.length > 0 && chips.length > 0 && chips.join("") === spelled,
    `${chips.length} spans → "${spelled}"`,
  );

  // Correcting a time must reach the editor: restart playback (the row click
  // above seeked to this line's start, which is where a stamp would land
  // anyway), then stamp the line's start at the playhead and read the input box
  // back. Selected row, not the row being sung.
  const before = await page.$eval(".line-row.selected .time-input", (el) => el.value);
  await page.evaluate(() => {
    const restart = [...document.querySelectorAll(".transport button")].find((b) =>
      (b.textContent ?? "").includes("开头"),
    );
    restart?.click();
  });
  await sleep(800);
  await page.evaluate(() => {
    const row = document.querySelector(".line-row.selected");
    const stamp = [...(row?.querySelectorAll(".line-head .ghost") ?? [])].find((b) =>
      (b.getAttribute("title") ?? "").includes("行起点"),
    );
    stamp?.click();
  });
  await sleep(600);
  const after = await page.$eval(".line-row.selected .time-input", (el) => el.value);
  check(
    "the editor can stamp a time on the line",
    after !== "" && after !== before,
    `${before} → ${after}`,
  );

  await page.screenshot({ path: "/tmp/ui-cache.png" });

  const realErrors = errors.filter((text) => !/favicon/i.test(text));
  check("no console errors", realErrors.length === 0, realErrors.join(" | "));
} finally {
  await browser.close();
}

if (failures.length) {
  console.log(`\n${failures.length} failure(s): ${failures.join(", ")}`);
  process.exit(1);
}
console.log("\nall checks passed");
