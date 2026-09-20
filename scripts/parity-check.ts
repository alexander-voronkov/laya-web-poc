// Token-for-token parity check of the ported sequence builder against the Python
// golden dump shipped with nvkudva/laya-web (app/public/parity.json).
//
// Usage (bundle first because Node cannot run the TS sources directly):
//   curl -o /tmp/parity.json \
//     https://raw.githubusercontent.com/nvkudva/laya-web/main/app/public/parity.json
//   npx esbuild scripts/parity-check.ts --bundle --platform=browser \
//     --format=esm --external:node:* --outfile=/tmp/pc.mjs
//   node /tmp/pc.mjs /tmp/parity.json
//
// Expects the model base to be reachable (fetches tokenizer.json from it).
import { readFileSync } from "node:fs";
import { loadTokenizer } from "../src/laya/tokenizer.js";
import { buildSequence, toInternal } from "../src/laya/sequence.js";

const MODELS_BASE =
  (process.env.VITE_MODELS_BASE as string | undefined) ||
  "https://huggingface.co/nvkudva/laya-web-q8/resolve/main/v1";

interface CaseDef {
  id: string;
  state: unknown;
  questions: Record<string, { type: "choice" | "score" | "noul"; instructions: string | unknown; criteria?: unknown }>;
  truncate_left?: boolean;
  expect: Record<string, { input_ids: number[]; marker_pos: number[] }>;
}

const path = process.argv[2];
if (!path) throw new Error("usage: node parity-check.mjs <parity.json>");
const parity = JSON.parse(readFileSync(path, "utf8")) as { cfg: { max_len: number; head_max_len: number }; cases: CaseDef[] };

const tok = await loadTokenizer(MODELS_BASE);
let fails = 0;
let total = 0;
for (const c of parity.cases) {
  for (const [qid, qdef] of Object.entries(c.questions)) {
    total++;
    const q = toInternal(qdef as never);
    const { ids, markers } = buildSequence(tok, c.state as never, q, parity.cfg.max_len, parity.cfg.head_max_len, c.truncate_left ?? false);
    const exp = c.expect[qid];
    const okIds = JSON.stringify(ids) === JSON.stringify(exp.input_ids);
    const okM = JSON.stringify(markers) === JSON.stringify(exp.marker_pos);
    if (!okIds || !okM) {
      fails++;
      console.log(`FAIL ${c.id}/${qid}: ids=${okIds} markers=${okM}`);
      if (!okIds) {
        for (let i = 0; i < Math.max(ids.length, exp.input_ids.length); i++) {
          if (ids[i] !== exp.input_ids[i]) {
            console.log(`  first diff at ${i}: got ${ids[i]}, expected ${exp.input_ids[i]}`);
            break;
          }
        }
      }
    }
  }
}
console.log(fails ? `${fails}/${total} cases FAILED` : `ALL ${total} cases PASS`);
process.exit(fails ? 1 : 0);
