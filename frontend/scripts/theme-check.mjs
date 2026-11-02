import puppeteer from "puppeteer-core";

const BASE = process.env.BASE ?? "http://127.0.0.1:8091";
const browser = await puppeteer.launch({
  executablePath: "/usr/bin/google-chrome",
  headless: true,
  args: ["--no-sandbox", "--disable-gpu", "--autoplay-policy=no-user-gesture-required"],
});
const page = await browser.newPage();
await page.setViewport({ width: 1440, height: 1000 });
const errors = [];
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
page.on("pageerror", (e) => errors.push(String(e)));

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

await page.goto(BASE, { waitUntil: "networkidle0" });
await wait(800);
console.log("default theme:", await page.evaluate(() => document.documentElement.dataset.theme));
console.log("bg:", await page.evaluate(() => getComputedStyle(document.body).backgroundColor));
await page.screenshot({ path: "/tmp/theme-light-list.png" });

// open the first project
const href = await page.$eval(".project-card a, .project-card .title a", (a) => a.getAttribute("href"));
await page.goto(`${BASE}${href}`, { waitUntil: "networkidle0" });
await wait(2500);
await page.screenshot({ path: "/tmp/theme-light-project.png", fullPage: false });

// switch to dark
await page.click(".theme-toggle");
await wait(600);
console.log("after toggle:", await page.evaluate(() => document.documentElement.dataset.theme));
console.log("bg:", await page.evaluate(() => getComputedStyle(document.body).backgroundColor));
console.log("stored:", await page.evaluate(() => localStorage.getItem("lyrictrace.theme")));
await page.screenshot({ path: "/tmp/theme-dark-project.png" });

// reload: preference should persist
await page.reload({ waitUntil: "networkidle0" });
await wait(1500);
console.log("after reload:", await page.evaluate(() => document.documentElement.dataset.theme));

// back to light and confirm it sticks across reload
await page.click(".theme-toggle");
await wait(400);
await page.reload({ waitUntil: "networkidle0" });
await wait(800);
console.log("back to light:", await page.evaluate(() => document.documentElement.dataset.theme));

// contrast sanity: dim text on card background
const contrast = await page.evaluate(() => {
  const lum = (rgb) => {
    const [r, g, b] = rgb.match(/\d+/g).slice(0, 3).map((v) => {
      const c = Number(v) / 255;
      return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
    });
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  };
  const ratio = (a, b) => {
    const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p);
    return +((x + 0.05) / (y + 0.05)).toFixed(2);
  };
  const card = document.querySelector(".card");
  const muted = document.querySelector(".muted") ?? document.body;
  return {
    cardBg: getComputedStyle(card).backgroundColor,
    text: getComputedStyle(document.body).color,
    muted: getComputedStyle(muted).color,
    textRatio: ratio(getComputedStyle(document.body).color, getComputedStyle(card).backgroundColor),
    mutedRatio: ratio(getComputedStyle(muted).color, getComputedStyle(card).backgroundColor),
  };
});
console.log(contrast);
console.log("console errors:", errors);
await browser.close();
