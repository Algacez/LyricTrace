import puppeteer from "puppeteer-core";

// The 「API 缓存」 page: it lists the newest /api/align jobs, plays one back and
// highlights the character being sung. BASE=http://127.0.0.1:8091 node scripts/ui-cache-smoke.mjs
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
  check("cache page keeps at most five", items <= 5, `${items} items`);

  const header = await page.$eval(".cache-item .title", (el) => el.textContent ?? "");
  check("a cached job is named", header.trim().length > 0, header.trim());

  const audio = await page.$eval("audio", (el) => el.getAttribute("src") ?? "");
  check("the source audio is served", audio.includes("/api/align/") && audio.endsWith("/audio"), audio);

  const lines = await page.$$eval(".cache-lines .line-row", (els) => els.length);
  check("the result renders its lines", lines > 0, `${lines} lines`);

  const words = await page.$$eval(".cache-words .karaoke-word", (els) => els.length);
  check("every character has its own span", words > 0, `${words} characters`);

  // Seeking to a line must move the player and light that character.
  await page.click(".cache-lines .line-row:nth-child(3)");
  await sleep(600);
  const active = await page.$$eval(".cache-lines .karaoke-word.active", (els) =>
    els.map((el) => el.textContent ?? ""),
  );
  check("seeking lights the character being sung", active.length === 1, JSON.stringify(active));

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
