import { TYPE_LABELS, type OptionItem, type QuestionItem } from "../questions";
import type { QuestionBudget } from "../budget";

const QUESTION_PLACEHOLDER: Record<string, string> = {
  noul: "Например: Is the text emotionally charged?",
  choice: "Например: What is the main topic of the text?",
  score: "Например: How formal is the tone of the text?",
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
  const set = <K extends keyof QuestionItem>(k: K, v: QuestionItem[K]) => onChange({ ...q, [k]: v });
  const isScore = q.type === "score";
  const items: OptionItem[] = isScore ? q.levels : q.options;
  const setItems = (next: OptionItem[]) => set(isScore ? "levels" : "options", next);

  return (
    <div className={`q-card${problem ? " invalid" : ""}`}>
      <div className="q-head">
        <span className="q-num">Вопрос {index + 1}</span>
        <span className="q-type">{TYPE_LABELS[q.type]}</span>
        <input
          className="q-id-input"
          value={q.id}
          title="Ключ, под которым ответ попадёт в JSON. Модели он не показывается."
          onChange={(e) => set("id", e.target.value)}
        />
        <div className="q-head-actions">
          <button className="icon-btn" title="Выше" disabled={index === 0} onClick={() => onMove(-1)}>↑</button>
          <button className="icon-btn" title="Ниже" disabled={index === total - 1} onClick={() => onMove(1)}>↓</button>
          <button className="icon-btn" title="Удалить вопрос" onClick={onRemove}>×</button>
        </div>
      </div>

      {problem && <div className="q-problem">{problem}</div>}

      <div className="field">
        <label>Вопрос</label>
        <input
          type="text"
          value={q.text}
          placeholder={QUESTION_PLACEHOLDER[q.type]}
          onChange={(e) => set("text", e.target.value)}
        />
      </div>

      <div className="field">
        <label>Уточнение (необязательно)</label>
        <input
          type="text"
          value={q.hint}
          placeholder="добавляется к формулировке вопроса"
          onChange={(e) => set("hint", e.target.value)}
        />
      </div>

      {q.type === "noul" && (
        <div className="field">
          <label>Что значит «да» и «нет» (необязательно)</label>
          <div className="crit-grid">
            <input
              type="text"
              value={q.criteriaTrue}
              placeholder="да — например: yes, the statement holds"
              onChange={(e) => set("criteriaTrue", e.target.value)}
            />
            <input
              type="text"
              value={q.criteriaFalse}
              placeholder="нет — например: no, it does not hold"
              onChange={(e) => set("criteriaFalse", e.target.value)}
            />
          </div>
          <div className="field-note">
            Эти две строки заменяют формулировки по умолчанию и стоят несколько токенов — при этом
            заметно повышают разделимость ответа.
          </div>
        </div>
      )}

      {q.type !== "noul" && (
        <div className="field">
          <label>{isScore ? "Уровни шкалы (по возрастанию)" : "Варианты ответа"}</label>
          <div className="opt-list">
            {items.map((o, j) => (
              <div className="opt-row" key={j}>
                <input
                  type="text"
                  className="opt-label"
                  value={o.label}
                  placeholder={isScore ? `уровень ${j}` : `вариант ${j + 1}`}
                  onChange={(e) => setItems(items.map((x, i) => (i === j ? { ...x, label: e.target.value } : x)))}
                />
                <input
                  type="text"
                  className="opt-desc"
                  value={o.description}
                  placeholder="описание (необязательно)"
                  onChange={(e) => setItems(items.map((x, i) => (i === j ? { ...x, description: e.target.value } : x)))}
                />
                <div className="opt-actions">
                  <button
                    className="icon-btn" title="Выше" disabled={j === 0}
                    onClick={() => { const n = [...items]; [n[j - 1], n[j]] = [n[j], n[j - 1]]; setItems(n); }}
                  >↑</button>
                  <button
                    className="icon-btn" title="Ниже" disabled={j === items.length - 1}
                    onClick={() => { const n = [...items]; [n[j + 1], n[j]] = [n[j], n[j + 1]]; setItems(n); }}
                  >↓</button>
                  <button
                    className="icon-btn" title="Удалить"
                    onClick={() => setItems(items.filter((_, i) => i !== j))}
                  >×</button>
                </div>
              </div>
            ))}
          </div>
          <button className="link-btn" onClick={() => setItems([...items, { label: "", description: "" }])}>
            {isScore ? "+ добавить уровень" : "+ добавить вариант"}
          </button>
          {isScore && (
            <div className="field-note">
              Порядок уровней задаёт шкалу: от низшего к высшему. Ожидание считается по индексам 0…n−1.
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
        токенов: {s.totalTokens} (вопрос {s.headTokens} · варианты {s.optionTokens} · текст {s.stateTokensUsed})
      </span>
      {budget.optionsDontFit && (
        <span className="error">варианты не помещаются в бюджет головы — вопрос не выполнится</span>
      )}
      {budget.instructionsClipped && (
        <span className="warn">
          формулировка обрезана: {s.headTokensFull} → {s.headTokens} токенов. Обрезается конец, то есть
          сам вопрос — сократите задачу или перенесите её в текст.
        </span>
      )}
      {budget.stateTruncated && (
        <span className="warn">
          текст обрезан: {s.stateTokens} → {s.stateTokensUsed} токенов, до модели дойдёт только начало.
        </span>
      )}
    </div>
  );
}
