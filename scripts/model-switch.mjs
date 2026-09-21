// Does the page answer with the model it names?
//
// This is the check that was missing. The builds differ in tuning, precision and
// calibration, so their answers to the same question differ visibly — which means a page
// serving the wrong one does not look broken, it looks like a different opinion.
// It shipped that way: selecting multilingual-fp16 downloaded english-q8 in full,
// english's boot resolved half a minute after the switch, and its callback wrote its own
// session into state because the liveness guard was a ref shared across effect runs.
// The page then answered with english under multilingual-fp16's name, to the decimal.
//
// Two assertions, and the first is the one that matters:
//   1. every weight actually downloaded belongs to the selected model
//   2. the answers differ from the default model's
//
// The second alone would not be enough — two models can agree by chance — and the first
// alone would not be either, since the app could fetch the right weights and still run a
// session built from the wrong ones. Together they pin it.
//
// Usage:
//   node scripts/model-switch.mjs                      # against production
//   SITE=http://localhost:4173 node scripts/model-switch.mjs
//   MODEL=multilingual node scripts/model-switch.mjs   # one model instead of all
import { chromium } from "playwright";

const SITE = process.env.SITE || "https://laya.voronkov.club";
/** Model id -> the folder it lives in inside the mirror.
 *
 *  Spelled out rather than derived by prefix, because prefixes lie here: "multilingual"
 *  is a prefix of "multilingual-fp16", so a prefix test would call fp16's weights the
 *  int8 model's and report a clean run for exactly the mix-up being looked for. The
 *  script checks this map against the picker and refuses to run if a model is missing,
 *  since a check that matches nothing reads the same as a check that passed. */
const FOLDER = {
  "multilingual-tuned-q8": "multilingual-tuned-q8",
  "multilingual-tuned": "multilingual-tuned-fp32-batched",
  "multilingual-fp16": "multilingual-fp16",
  "multilingual": "multilingual-int8",
};
// The default is what boots before anything is picked, so it is the one that can bleed
// into another model's run. Every other model is checked against it.
const DEFAULT_MODEL = process.env.DEFAULT_MODEL || "multilingual-tuned-q8";
const MODELS = (process.env.MODEL || "multilingual-tuned,multilingual-fp16,multilingual").split(",");

const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

/** Load one model in a fresh profile, run the seeded questions, report the bytes. */
async function runOne(browser, model) {
  const ctx = await browser.newContext({ viewport: { width: 1400, height: 1200 } });
  const page = await ctx.newPage();
  // Bytes per variant folder, counted from finished responses only.
  //
  // Counting requests instead reports a false failure every time: the page boots the
  // default model on mount, the switch arrives a second later, and the abandoned
  // download leaves a request behind that was correctly cancelled. What matters is not
  // whether another model's URL was touched, it is whether its weights were paid for --
  // so this measures transferred bytes and calls anything under a megabyte the tail of
  // a cancelled boot rather than a download.
  const bytes = new Map();
  const errs = [];
  // The folder is on the *first* URL of a redirect chain, not the last. The Hub answers
  // /laya-web/resolve/main/<folder>/<file> with a redirect to a CDN host whose path says
  // nothing about which variant it belongs to, and the bytes arrive on that second
  // request. Matching only the URL that finished therefore attributed every download to
  // nothing at all, and the check reported a clean run having measured zero -- which is
  // the failure mode this file's own comments warn about, walked into while writing them.
  const folderOf = (req) => {
    for (let r = req; r; r = r.redirectedFrom()) {
      const m = r.url().match(/\/laya-web\/resolve\/main\/([^/]+)\//);
      if (m) return m[1];
    }
    return undefined;
  };
  page.on("requestfinished", async (r) => {
    const folder = folderOf(r);
    if (!folder) return;
    try {
      const sizes = await r.sizes();
      bytes.set(folder, (bytes.get(folder) ?? 0) + (sizes.responseBodySize ?? 0));
    } catch {
      // The request outlived its context; the byte count is best-effort by nature.
    }
  });
  page.on("pageerror", (e) => errs.push(String(e).slice(0, 200)));

  await page.goto(SITE, { waitUntil: "domcontentloaded", timeout: 60_000 });
  await page.evaluate(() => localStorage.clear());
  await page.reload({ waitUntil: "domcontentloaded" });
  await page.waitForSelector("#model-picker", { timeout: 60_000 });
  await page.selectOption("#model-picker", model);

  const t0 = Date.now();
  await page.waitForSelector("#overlay.hidden, .run-card button.primary:not([disabled]), #ov-status.error", { timeout: 900_000 });
  const refusal = await page.evaluate(() =>
    document.querySelector("#ov-status.error")?.textContent?.trim() ?? null);
  if (refusal) {
    await ctx.close();
    // A build that refuses is not a failure of this check: multilingual-fp16 says so
    // outright on a browser with no WebGPU, and saying so is correct behaviour.
    return { model, refused: refusal, bytes: Object.fromEntries(bytes) };
  }
  log(`  ${model}: loaded in ${Math.round((Date.now() - t0) / 1000)}s`);

  // Clicked from inside the page: inference runs on the main thread, so Playwright's
  // click would wait for a page that cannot settle until the whole run is finished.
  await page.evaluate(() => document.querySelector(".run-card button.btn.primary")?.click());
  await page.waitForFunction(
    () => /done in|stopped:|failed on/.test(document.querySelector(".run-card .run-status")?.textContent ?? ""),
    { timeout: 600_000 });

  const answers = await page.evaluate(() =>
    [...document.querySelectorAll(".ans-card")].map((c) =>
      (c.querySelector(".noul-big") ?? c.querySelector(".ans-sub"))?.textContent?.trim()));
  await ctx.close();
  return { model, answers, bytes: Object.fromEntries(bytes), errs };
}

const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan"] });
const failures = [];
try {
  // Every model the picker offers must be in FOLDER, or a new build would be checked
  // against nothing and pass.
  {
    const page = await browser.newPage();
    await page.goto(SITE, { waitUntil: "domcontentloaded", timeout: 60_000 });
    await page.waitForSelector("#model-picker", { timeout: 60_000 });
    const offered = await page.$$eval("#model-picker option", (os) => os.map((o) => o.value));
    await page.close();
    const unknown = offered.filter((o) => !(o in FOLDER));
    if (unknown.length) throw new Error(`the picker offers models this script cannot place: ${unknown.join(", ")}`);
    console.log(`picker offers ${offered.length} models, all known`);
  }

  log(`baseline: ${DEFAULT_MODEL}`);
  const base = await runOne(browser, DEFAULT_MODEL);
  if (base.refused) throw new Error(`the default model refused to load: ${base.refused}`);
  console.log(`  answers: ${JSON.stringify(base.answers)}`);

  for (const model of MODELS) {
    log(`checking: ${model}`);
    const r = await runOne(browser, model);
    if (r.refused) {
      console.log(`  refused, which is a valid outcome: ${r.refused}`);
      continue;
    }
    // A megabyte is the line between "a cancelled boot left a few chunks behind" and
    // "this page downloaded a model it was not asked for". The smallest real weight file
    // here is the 34 MB tokenizer, so nothing legitimate lands between the two.
    const STRAY_BYTES = 1_000_000;
    const own = r.bytes[FOLDER[model]] ?? 0;
    const foreign = Object.entries(r.bytes)
      .filter(([f, n]) => f !== FOLDER[model] && n > STRAY_BYTES)
      .map(([f, n]) => `${f} (${(n / 1e6).toFixed(1)} MB)`);
    // Before believing "no foreign bytes", prove the meter works. A model that loaded
    // from an empty cache must show most of itself arriving; if it shows nothing, the
    // accounting is broken and a clean result means only that nothing was counted.
    const EXPECT_OWN_BYTES = 50_000_000;
    if (own < EXPECT_OWN_BYTES) {
      failures.push(`${model}: only ${(own / 1e6).toFixed(1)} MB attributed to its own weights — ` +
        "the byte accounting is not working, so this run proves nothing");
      console.log(`  FAIL — measured ${(own / 1e6).toFixed(1)} MB of its own weights; the meter is broken`);
    }
    const sameAsBase = JSON.stringify(r.answers) === JSON.stringify(base.answers);

    const summary = Object.entries(r.bytes)
      .map(([f, n]) => `${f} ${(n / 1e6).toFixed(1)} MB`)
      .join(", ");
    console.log(`  downloaded: ${summary || "(nothing — served from cache?)"}`);
    console.log(`  answers: ${JSON.stringify(r.answers)}`);
    if (foreign.length) {
      failures.push(`${model}: downloaded another model's weights — ${foreign.join(", ")}`);
      console.log(`  FAIL — downloaded ${foreign.join(", ")}`);
    }
    if (sameAsBase) {
      failures.push(`${model}: answered identically to ${DEFAULT_MODEL}, so it is probably not the model that ran`);
      console.log(`  FAIL — answers identical to ${DEFAULT_MODEL}`);
    }
    if (!foreign.length && !sameAsBase) console.log("  ok");
    if (r.errs?.length) console.log(`  console errors: ${r.errs.join(" | ")}`);
  }
} finally {
  await browser.close();
}

if (failures.length) {
  console.error("\n" + failures.length + " failure(s):");
  for (const f of failures) console.error("  " + f);
  process.exit(1);
}
console.log("\nevery model answered with its own weights");
