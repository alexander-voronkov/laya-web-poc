// Does the page answer with the model it names?
//
// This is the check that was missing. The four builds differ by architecture, tokenizer
// and calibration, so their answers to the same question differ visibly — which means a
// page serving the wrong one does not look broken, it looks like a different opinion.
// It shipped that way: selecting multilingual-fp16 downloaded english-q8 in full,
// english's boot resolved half a minute after the switch, and its callback wrote its own
// session into state because the liveness guard was a ref shared across effect runs.
// The page then answered with english under multilingual-fp16's name, to the decimal.
//
// Two assertions, and the first is the one that matters:
//   1. every weight fetched belongs to the selected model, and none to any other
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
  "english": "english-q8",
  "typed-decisions": "typed-decisions-q8",
  "multilingual": "multilingual-int8",
  "multilingual-fp16": "multilingual-fp16",
};
// The default is what boots before anything is picked, so it is the one that can bleed
// into another model's run. Every other model is checked against it.
const DEFAULT_MODEL = process.env.DEFAULT_MODEL || "english";
const MODELS = (process.env.MODEL || "typed-decisions,multilingual,multilingual-fp16").split(",");

const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

/** Load one model in a fresh profile, run the seeded questions, report what it fetched. */
async function runOne(browser, model) {
  const ctx = await browser.newContext({ viewport: { width: 1400, height: 1200 } });
  const page = await ctx.newPage();
  const fetched = new Set();
  const errs = [];
  page.on("request", (r) => {
    const m = r.url().match(/\/laya-web\/resolve\/main\/([^/]+)\//);
    if (m) fetched.add(m[1]);
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
    return { model, refused: refusal, fetched: [...fetched] };
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
  return { model, answers, fetched: [...fetched], errs };
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
    const foreign = r.fetched.filter((f) => f !== FOLDER[model]);
    const sameAsBase = JSON.stringify(r.answers) === JSON.stringify(base.answers);

    console.log(`  fetched from: ${r.fetched.join(", ") || "(nothing — served from cache?)"}`);
    console.log(`  answers: ${JSON.stringify(r.answers)}`);
    if (foreign.length) {
      failures.push(`${model}: fetched another model's weights (${foreign.join(", ")})`);
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
