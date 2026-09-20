// The request builder, the budget analysis and the answer cards have to agree on one
// thing: which options exist and in what order. When they disagreed, nothing crashed
// and nothing was reported -- a blank row became a scored option the cards hid, and for
// `score`, where the model answers by position, every label ended up beside its
// neighbour's probability. These tests pin that agreement down. No network needed.
import { describe, expect, it } from "vitest";
import {
  buildInstructions,
  buildState,
  countOptions,
  newQuestion,
  scoredItems,
  toQuestionDef,
  toRequest,
  validateQuestions,
  type QuestionItem,
} from "../src/questions";

const q = (p: Partial<QuestionItem> & Pick<QuestionItem, "type">): QuestionItem => ({
  ...newQuestion(p.type, 1),
  text: "Does it hold?",
  ...p,
});

describe("blank rows never reach the model", () => {
  it("drops an unlabelled choice option", () => {
    const item = q({
      type: "choice",
      options: [
        { label: "ironic", description: "" },
        { label: "  ", description: "a description with no label" },
        { label: "earnest", description: "" },
      ],
    });
    const def = toQuestionDef("", item, "instructions");
    expect(Object.keys(def.criteria as object)).toEqual(["ironic", "earnest"]);
    // The empty string must not survive as a key: it would be a real scored option
    // rendered as `": "`, and two of them would silently collapse into one.
    expect(Object.keys(def.criteria as object)).not.toContain("");
    expect(countOptions(item)).toBe(2);
  });

  it("drops an unlabelled score level, so display indices match answer indices", () => {
    const item = q({
      type: "score",
      levels: [
        { label: "", description: "" },
        { label: "medium", description: "" },
        { label: "high", description: "" },
      ],
    });
    const criteria = toQuestionDef("", item, "instructions").criteria as string[];
    expect(criteria).toEqual(["medium", "high"]);
    // The model answers under "0" and "1"; the cards render scoredItems in this order.
    // If these two lists ever diverge again, the labels shift by one.
    expect(scoredItems(item).map((l) => l.label)).toEqual(criteria);
  });

  it("flags blank rows rather than dropping them quietly", () => {
    const item = q({
      type: "choice",
      options: [
        { label: "a", description: "" },
        { label: "b", description: "" },
        { label: "", description: "" },
      ],
    });
    const problems = validateQuestions([item]).map((i) => i.problem);
    expect(problems.some((p) => p.includes("unlabelled"))).toBe(true);
  });
});

describe("validation catches keys that would corrupt the response", () => {
  it("rejects an empty key", () => {
    const problems = validateQuestions([q({ type: "noul", id: "" })]).map((i) => i.problem);
    expect(problems).toContain("the question key is empty");
  });

  it("rejects duplicate keys, which would collapse in the request object", () => {
    const items = [q({ type: "noul", id: "same" }), q({ type: "noul", id: "same" })];
    expect(Object.keys(toRequest("", items, "instructions"))).toHaveLength(1);
    expect(validateQuestions(items).map((i) => i.problem)).toContain("the question key is used twice");
  });

  it("reports against the stable uid, not the editable key", () => {
    const item = q({ type: "noul", id: "", text: "" });
    for (const issue of validateQuestions([item])) expect(issue.uid).toBe(item.uid);
  });
});

describe("noul criteria", () => {
  it("omits absent sides rather than sending empty strings", () => {
    const def = toQuestionDef("", q({ type: "noul", criteriaTrue: "yes it does", criteriaFalse: "  " }), "instructions");
    // An absent key is what the Python reference produces, and it is the shape the
    // golden dump covers; an empty one takes a different branch in renderOptions.
    expect(def.criteria).toEqual({ true: "yes it does" });
  });

  it("is null when neither side is given", () => {
    expect(toQuestionDef("", q({ type: "noul" }), "instructions").criteria).toBeNull();
  });
});

describe("task framing placement", () => {
  const item = q({ type: "noul", text: "Is it ironic?" });

  it("goes into the instructions only in that mode", () => {
    expect(buildInstructions("Read as a critic", item, "instructions")).toBe("Read as a critic. Is it ironic?");
    expect(buildInstructions("Read as a critic", item, "state")).toBe("Is it ironic?");
    expect(buildInstructions("Read as a critic", item, "both")).toBe("Read as a critic. Is it ironic?");
  });

  it("goes into the state only in that mode", () => {
    expect(buildState("Read as a critic", "Some text.", "state")).toBe("Read as a critic\n\n---\n\nSome text.");
    expect(buildState("Read as a critic", "Some text.", "instructions")).toBe("Some text.");
  });

  it("does not leave a dangling separator when one side is empty", () => {
    expect(buildState("Read as a critic", "", "state")).toBe("Read as a critic");
    expect(buildState("", "Some text.", "state")).toBe("Some text.");
  });
});
