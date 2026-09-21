// Batching must be a speed change, not an answer change.
//
// Kept although batching is currently reverted: see README, "Why questions are not
// batched". This is the harness that caught the export refusing batch>1 on the first
// attempt, and it is what any future attempt has to pass.
//
// Needs ?batchTokens support in the app, which lives with the batching code.
//
// Runs the seeded questions twice on the live site — once with ?batchTokens=1, which
// puts every question in its own forward pass, and once with the default batching —
// and compares the rendered distributions. A speed-up that quietly moves probabilities
// is not a speed-up, it is a different model.
import { chromium } from "playwright";

const SITE = process.env.SITE || "https://laya.voronkov.club";
const ctx = await chromium.launchPersistentContext(process.env.PROFILE || "./.chrome-profile");
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

async function run(page, url, label) {
  await page.goto(url, { waitUntil: "domcontentloaded", timeout: 60_000 });
  await page.evaluate(() => localStorage.clear());
  await page.reload({ waitUntil: "domcontentloaded" });
  await page.waitForSelector(".run-card button.btn.primary:not([disabled])", { timeout: 900_000 });

  const t0 = Date.now();
  await page.evaluate(() => document.querySelector(".run-card button.btn.primary").click());
  await page.waitForFunction(
    () => /done in|failed on/.test(document.querySelector(".run-card span")?.textContent ?? ""),
    { timeout: 900_000 },
  );
  const wall = Date.now() - t0;

  const out = await page.evaluate(() => ({
    status: document.querySelector(".run-card span")?.textContent?.trim(),
    passes: [...document.querySelectorAll(".metrics-rows .m-row")]
      .map((r) => r.textContent.replace(/\s+/g, " ").trim())
      .find((t) => t.startsWith("Forward passes")),
    answers: [...document.querySelectorAll(".ans-card")].map((c) => ({
      q: c.querySelector(".ans-text")?.textContent,
      // Every number the card shows, in order — the comparison should not depend on
      // which primitive produced it.
      values: [
        c.querySelector(".noul-big")?.textContent,
        ...[...c.querySelectorAll(".dist-pct")].map((p) => p.textContent),
      ].filter(Boolean),
    })),
  }));
  log(`${label}: ${out.status}  (${(wall / 1000).toFixed(1)}s wall)`);
  log(`  ${out.passes}`);
  return { ...out, wall };
}

try {
  const page = ctx.pages()[0] ?? (await ctx.newPage());
  const seq = await run(page, `${SITE}/?batchTokens=1`, "one pass per question");
  const bat = await run(page, `${SITE}/`, "batched            ");

  console.log("\n================ ANSWERS ================");
  let same = true;
  for (let i = 0; i < Math.max(seq.answers.length, bat.answers.length); i++) {
    const a = seq.answers[i], b = bat.answers[i];
    const equal = JSON.stringify(a?.values) === JSON.stringify(b?.values);
    same &&= equal;
    console.log(`${equal ? "same" : "DIFF"}  ${a?.q ?? b?.q}`);
    console.log(`      sequential: ${a?.values?.join("  ")}`);
    console.log(`      batched   : ${b?.values?.join("  ")}`);
  }

  console.log("\n================ VERDICT ================");
  console.log(same
    ? "Identical distributions — batching changed the speed and nothing else."
    : "DIFFERENT distributions — batching is altering the answers. Do not ship.");
  console.log(`wall clock: ${(seq.wall / 1000).toFixed(1)}s sequential -> ${(bat.wall / 1000).toFixed(1)}s batched` +
    (bat.wall > 0 ? `  (${(seq.wall / bat.wall).toFixed(2)}x)` : ""));
} finally {
  await ctx.close();
}
