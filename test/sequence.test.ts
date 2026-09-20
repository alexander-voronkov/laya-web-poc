// The one claim this whole prototype rests on: the TypeScript sequence builder emits
// exactly the token ids the Python reference emits. If it drifts by a single token,
// every probability on the page is still a plausible-looking number -- just a number
// about a different question. Nothing downstream would notice.
//
// So this is not a smoke test. The fixture is the golden dump shipped with
// nvkudva/laya-web, produced by running the real Python implementation, and it is
// vendored rather than fetched so a network hiccup cannot turn into a silent skip.
// 24 cases / 26 questions covering both truncation branches, option-budget shrinking,
// non-Latin script, mask-token injection and degenerate inputs.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it, beforeAll } from "vitest";
import { buildSequence, renderOptions, toInternal } from "../src/laya/sequence";
import { loadTokenizer } from "../src/laya/tokenizer";
import type { QuestionDef, State } from "../src/laya/types";
import type { Tok } from "../src/laya/sequence";

const MODELS_BASE =
  process.env.VITE_MODELS_BASE ||
  "https://huggingface.co/nvkudva/laya-web-q8/resolve/main/v1";

interface Expected {
  input_ids: number[];
  marker_pos: number[];
  rendered_options: string[];
}

interface Case {
  id: string;
  state: State;
  questions: Record<string, QuestionDef>;
  truncate_left?: boolean;
  expect: Record<string, Expected>;
}

const fixture = JSON.parse(
  readFileSync(fileURLToPath(new URL("./fixtures/parity.json", import.meta.url)), "utf8"),
) as { cfg: { max_len: number; head_max_len: number }; cases: Case[] };

let tok: Tok;

beforeAll(async () => {
  tok = await loadTokenizer(MODELS_BASE);
});

describe("buildSequence parity with the Python reference", () => {
  // One test per case, so a failure names the input rather than "expected [12,7,…]".
  for (const c of fixture.cases) {
    it(c.id, () => {
      for (const [qid, qdef] of Object.entries(c.questions)) {
        const q = toInternal(qdef);
        const expected = c.expect[qid];
        const { ids, markers, stats } = buildSequence(
          tok, c.state, q, fixture.cfg.max_len, fixture.cfg.head_max_len, c.truncate_left ?? false,
        );

        expect(renderOptions(q), `${c.id}/${qid} rendered options`).toEqual(expected.rendered_options);
        expect(ids, `${c.id}/${qid} input_ids`).toEqual(expected.input_ids);
        expect(markers, `${c.id}/${qid} marker_pos`).toEqual(expected.marker_pos);

        // The stats block is the laya-web-poc addition. It must describe the sequence
        // that was actually produced -- a stats field that disagrees with the ids would
        // put wrong numbers on the metrics panel, which is the opposite of its job.
        expect(stats.totalTokens, `${c.id}/${qid} stats.totalTokens`).toBe(ids.length);
        expect(stats.stateTokensUsed).toBeLessThanOrEqual(stats.stateTokens);
        expect(stats.headTokens).toBeLessThanOrEqual(stats.headTokensFull);
        expect(stats.optionTokens).toBeLessThanOrEqual(stats.optionTokensFull);
        expect(stats.totalTokens).toBeLessThanOrEqual(fixture.cfg.max_len);
      }
    });
  }

  it("covers every question in the fixture", () => {
    const n = fixture.cases.reduce((a, c) => a + Object.keys(c.questions).length, 0);
    // A fixture that silently shrank to nothing would pass every assertion above.
    expect(n).toBe(26);
  });
});
