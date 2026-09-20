// Browser smoke check for https://laya.voronkov.club
//
// The one thing CI cannot prove: that the 524MB of weights download, the wasm
// session builds, and a forward pass produces probabilities. Everything up to
// inference is gated; this is the part that is not.
//
// Usage:
//   node smoke.mjs --local                          # local Chromium
//   PLAYWRIGHT_WS=ws://host:9223/ node smoke.mjs     # a remote Playwright server
//   SITE=http://localhost:4173 node smoke.mjs --local
import { chromium } from "playwright";

const SITE = process.env.SITE || "https://laya.voronkov.club";
const local = process.argv.includes("--local");
const REMOTE = process.env.PLAYWRIGHT_WS;
if (!local && !REMOTE) {
  console.error("set PLAYWRIGHT_WS to a Playwright server, or pass --local");
  process.exit(2);
}

const consoleErrs = [];
const failedReqs = [];

const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

const browser = local
  ? await chromium.launch()
  : await chromium.connect(REMOTE);

try {
  const context = await browser.newContext({ viewport: { width: 1400, height: 1400 } });
  const page = await context.newPage();

  page.on("console", (m) => { if (m.type() === "error") consoleErrs.push(m.text()); });
  page.on("pageerror", (e) => consoleErrs.push("PAGEERROR: " + String(e)));
  page.on("requestfailed", (r) => failedReqs.push(`${r.failure()?.errorText} ${r.url().slice(0, 120)}`));

  log("opening", SITE);
  await page.goto(SITE, { waitUntil: "domcontentloaded", timeout: 60_000 });

  // Cross-origin isolation is what decides whether this runs on 8 threads or 1.
  const isolated = await page.evaluate(() => crossOriginIsolated);
  const sab = await page.evaluate(() => typeof SharedArrayBuffer !== "undefined");
  log("crossOriginIsolated:", isolated, "| SharedArrayBuffer:", sab);

  // The overlay stays up until both ONNX sessions exist. 524MB over an unknown
  // link, so this is the long wait; report progress rather than sitting mute.
  const started = Date.now();
  const tick = setInterval(async () => {
    try {
      const status = await page.locator("#ov-status").textContent({ timeout: 2000 });
      const sizes = await page.locator(".ov-size").allTextContents();
      log(`  ${Math.round((Date.now() - started) / 1000)}s  ${status?.trim()}  [${sizes.join(" | ")}]`);
    } catch { /* overlay gone */ }
  }, 15_000);

  await page.waitForSelector("#overlay.hidden, .run-card button.primary:not([disabled])", { timeout: 900_000 });
  clearInterval(tick);
  log(`model ready after ${Math.round((Date.now() - started) / 1000)}s`);

  await page.screenshot({ path: "shot-loaded.png", fullPage: true });

  log("running inference…");
  const t0 = Date.now();
  await page.getByRole("button", { name: "Get answers" }).click();
  // Three seeded questions, one forward pass each, on the main thread.
  await page.waitForSelector(".ans-card", { timeout: 300_000 });
  await page.waitForFunction(
    () => /done in|stopped:|failed on/.test(document.querySelector(".run-card span")?.textContent ?? ""),
    { timeout: 300_000 },
  );
  log(`run finished in ${Math.round((Date.now() - t0) / 1000)}s`);

  const answers = await page.evaluate(() =>
    [...document.querySelectorAll(".ans-card")].map((c) => ({
      question: c.querySelector(".ans-text")?.textContent,
      headline: c.querySelector(".noul-big")?.textContent ?? c.querySelector(".ans-sub")?.textContent,
      dist: [...c.querySelectorAll(".dist-row")].map((r) => r.textContent?.replace(/\s+/g, " ").trim()),
      ms: c.querySelector(".ans-ms")?.textContent,
    })),
  );
  const metrics = await page.evaluate(() =>
    [...document.querySelectorAll(".metrics-rows .m-row")].map((r) => r.textContent?.replace(/\s+/g, " ").trim()),
  );
  const table = await page.evaluate(() =>
    [...document.querySelectorAll(".metrics-table tr")].map((r) =>
      [...r.children].map((c) => c.textContent?.trim()).join(" | "),
    ),
  );
  const status = await page.locator(".run-card span").textContent();

  await page.screenshot({ path: "shot-answers.png", fullPage: true });

  console.log("\n================ STATUS ================\n" + status);
  console.log("\n================ ANSWERS ================");
  console.log(JSON.stringify(answers, null, 2));
  console.log("\n================ METRICS ================");
  for (const m of metrics) console.log("  " + m);
  console.log("\n================ PER QUESTION ================");
  for (const t of table) console.log("  " + t);
  console.log("\n================ CONSOLE ERRORS ================");
  console.log(consoleErrs.length ? consoleErrs.join("\n") : "  (none)");
  console.log("\n================ FAILED REQUESTS ================");
  console.log(failedReqs.length ? failedReqs.join("\n") : "  (none)");
} finally {
  await browser.close();
}
