import { TYPE_LABELS, type OptionItem, type QuestionItem } from "../questions";
import type { QuestionBudget } from "../budget";

const QUESTION_PLACEHOLDER: Record<string, string> = {
  noul: "e.g. Is the text emotionally charged?",
  choice: "e.g. What is the main topic of the text?",
  score: "e.g. How formal is the tone of the text?",
};

interface Props {
  question: QuestionItem;
  index: number;
  total: number;
  problem?: string;
  advice: string[];
  budget?: QuestionBudget;
  onChange: (q: QuestionItem) => void;
  onRemove: () => void;
  onMove: (delta: number) => void;
}

export function QuestionCard({
  question: q, index, total, problem, advice, budget, onChange, onRemove, onMove,
}: Props) {
  const set = (patch: Partial<QuestionItem>) => onChange({ ...q, ...patch });
  const isScore = q.type === "score";
  const items: OptionItem[] = isScore ? q.levels : q.options;
  const setItems = (next: OptionItem[]) => set(isScore ? { levels: next } : { options: next });

  return (
    <div className={`q-card${problem ? " invalid" : ""}`}>
      <div className="q-head">
        <span className="q-num">Question {index + 1}</span>
        <span className="q-type">{TYPE_LABELS[q.type]}</span>
        <input
          className="q-id-input"
          value={q.id}
          title="The key this answer appears under in the export. Never shown to the model."
          onChange={(e) => set({ id: e.target.value })}
        />
        <div className="q-head-actions">
          <button className="icon-btn" title="Move up" disabled={index === 0} onClick={() => onMove(-1)}>↑</button>
          <button className="icon-btn" title="Move down" disabled={index === total - 1} onClick={() => onMove(1)}>↓</button>
          <button className="icon-btn" title="Remove question" onClick={onRemove}>×</button>
        </div>
      </div>

      {problem && <div className="q-problem">{problem}</div>}

      <div className="field">
        <label>Question</label>
        <input
          type="text"
          value={q.text}
          placeholder={QUESTION_PLACEHOLDER[q.type]}
          onChange={(e) => set({ text: e.target.value })}
        />
      </div>

      <div className="field">
        <label>Hint (optional)</label>
        <input
          type="text"
          value={q.hint}
          placeholder="appended to the question wording"
          onChange={(e) => set({ hint: e.target.value })}
        />
      </div>

      {q.type === "noul" && (
        <div className="field">
          <label>What yes and no mean (optional)</label>
          <div className="crit-grid">
            <input
              type="text"
              value={q.criteriaTrue}
              placeholder="yes — e.g. yes, the statement holds"
              onChange={(e) => set({ criteriaTrue: e.target.value })}
            />
            <input
              type="text"
              value={q.criteriaFalse}
              placeholder="no — e.g. no, it does not hold"
              onChange={(e) => set({ criteriaFalse: e.target.value })}
            />
          </div>
          <div className="field-note">
            These two lines replace the default wording. They cost a handful of tokens and
            measurably sharpen the answer.
          </div>
        </div>
      )}

      {q.type !== "noul" && (
        <div className="field">
          <label>{isScore ? "Scale levels, low to high" : "Options"}</label>
          <div className="opt-list">
            {items.map((o, j) => (
              <div className="opt-row" key={j}>
                <input
                  type="text"
                  className="opt-label"
                  value={o.label}
                  placeholder={isScore ? `level ${j}` : `option ${j + 1}`}
                  onChange={(e) => setItems(items.map((x, i) => (i === j ? { ...x, label: e.target.value } : x)))}
                />
                <input
                  type="text"
                  className="opt-desc"
                  value={o.description}
                  placeholder="description (optional)"
                  onChange={(e) => setItems(items.map((x, i) => (i === j ? { ...x, description: e.target.value } : x)))}
                />
                <div className="opt-actions">
                  <button
                    className="icon-btn" title="Move up" disabled={j === 0}
                    onClick={() => { const n = [...items]; [n[j - 1], n[j]] = [n[j], n[j - 1]]; setItems(n); }}
                  >↑</button>
                  <button
                    className="icon-btn" title="Move down" disabled={j === items.length - 1}
                    onClick={() => { const n = [...items]; [n[j + 1], n[j]] = [n[j], n[j + 1]]; setItems(n); }}
                  >↓</button>
                  <button
                    className="icon-btn" title="Remove"
                    onClick={() => setItems(items.filter((_, i) => i !== j))}
                  >×</button>
                </div>
              </div>
            ))}
          </div>
          <button className="link-btn" onClick={() => setItems([...items, { label: "", description: "" }])}>
            {isScore ? "+ add level" : "+ add option"}
          </button>
          {isScore && (
            <div className="field-note">
              The order defines the scale, lowest to highest. The expected value is computed
              over the indices 0…n−1.
            </div>
          )}
        </div>
      )}

      {budget && <BudgetLine budget={budget} />}
      {advice.map((a, i) => <div className="q-advice" key={i}>{a}</div>)}
    </div>
  );
}

/** What this question will cost, before it is run. The two failure lines are the
 *  point: both are silent in the model's output, and both change the answer. */
function BudgetLine({ budget }: { budget: QuestionBudget }) {
  const s = budget.stats;
  return (
    <div className="q-budget">
      <span className="muted">
        tokens: {s.totalTokens} (question {s.headTokens} · options {s.optionTokens} · text {s.stateTokensUsed})
      </span>
      {budget.optionsDontFit && (
        <span className="error">the options do not fit the head budget — this question cannot run</span>
      )}
      {budget.optionsShrunk && (
        <span className="warn">
          options shortened: {s.optionTokensFull} → {s.optionTokens} tokens. Each option is cut
          individually and mid-word, so the model scores the mangled text. Shorten the descriptions
          or use fewer options.
        </span>
      )}
      {budget.instructionsClipped && (
        <span className="warn">
          wording clipped: {s.headTokensFull} → {s.headTokens} tokens. The cut lands on the end,
          which is the question itself — shorten the task or move it into the text.
        </span>
      )}
      {budget.stateTruncated && (
        <span className="warn">
          text truncated: {s.stateTokens} → {s.stateTokensUsed} tokens; only the beginning reaches
          the model.
        </span>
      )}
    </div>
  );
}
