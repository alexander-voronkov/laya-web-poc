import { h } from "./dom";
import { TYPE_LABELS, type OptionItem, type QuestionItem } from "./state";

export interface QuestionEvents {
  /** Structural change (add/remove/reorder) -> full re-render of the list. */
  onChanged(): void;
}

const QUESTION_PLACEHOLDER: Record<string, string> = {
  noul: "Например: Is the text emotionally charged?",
  choice: "Например: What is the main topic of the text?",
  score: "Например: How formal is the tone of the text?",
};

export function renderQuestions(
  root: HTMLElement,
  questions: QuestionItem[],
  ev: QuestionEvents,
  invalid: Map<string, string>,
): void {
  root.replaceChildren();
  if (!questions.length) {
    root.append(h("div", { class: "empty-note" }, "Вопросов пока нет — добавьте первый ниже."));
    return;
  }
  questions.forEach((q, i) => root.append(questionCard(q, i, questions, ev, invalid.get(q.id))));
}

function textInput(value: string, oninput: (v: string) => void, placeholder: string, cls?: string) {
  return h("input", {
    type: "text",
    class: cls,
    value,
    placeholder,
    oninput: (e: Event) => oninput((e.target as HTMLInputElement).value),
  });
}

function optionRow(
  o: OptionItem,
  j: number,
  opts: OptionItem[],
  rerender: () => void,
  isScore: boolean,
): HTMLElement {
  const swap = (a: number, b: number) => {
    const t = opts[a];
    opts[a] = opts[b];
    opts[b] = t;
    rerender();
  };
  return h(
    "div",
    { class: "opt-row" },
    textInput(o.label, (v) => (o.label = v), isScore ? `уровень ${j}` : `вариант ${j + 1}`, "opt-label"),
    textInput(o.description, (v) => (o.description = v), "описание (необязательно)", "opt-desc"),
    h(
      "div",
      { class: "opt-actions" },
      h("button", { class: "icon-btn", title: "Выше", disabled: j === 0 ? true : null, onclick: () => swap(j, j - 1) }, "↑"),
      h("button", { class: "icon-btn", title: "Ниже", disabled: j === opts.length - 1 ? true : null, onclick: () => swap(j, j + 1) }, "↓"),
      h("button", { class: "icon-btn", title: "Удалить", onclick: () => { opts.splice(j, 1); rerender(); } }, "×"),
    ),
  );
}

function questionCard(
  q: QuestionItem,
  i: number,
  questions: QuestionItem[],
  ev: QuestionEvents,
  problem: string | undefined,
): HTMLElement {
  const card = h("div", { class: `q-card${problem ? " invalid" : ""}` });
  card.append(
    h(
      "div",
      { class: "q-head" },
      h("span", { class: "q-num" }, `Вопрос ${i + 1}`),
      h("span", { class: "q-type" }, TYPE_LABELS[q.type]),
      h("span", { class: "q-id" }, q.id),
      h("button", { class: "icon-btn", title: "Удалить вопрос", onclick: () => { questions.splice(i, 1); ev.onChanged(); } }, "×"),
    ),
  );
  if (problem) card.append(h("div", { class: "q-problem" }, problem));
  card.append(
    h("div", { class: "field" },
      h("label", null, "Вопрос"),
      textInput(q.text, (v) => (q.text = v), QUESTION_PLACEHOLDER[q.type]),
    ),
    h("div", { class: "field" },
      h("label", null, "Уточнение (необязательно)"),
      textInput(q.hint, (v) => (q.hint = v), "добавляется к формулировке вопроса"),
    ),
  );
  if (q.type !== "noul") {
    const isScore = q.type === "score";
    const opts = isScore ? q.levels : q.options;
    const list = h("div", { class: "opt-list" });
    const rerender = () => {
      list.replaceChildren(...opts.map((o, j) => optionRow(o, j, opts, rerender, isScore)));
    };
    rerender();
    card.append(
      h("div", { class: "field" },
        h("label", null, isScore ? "Уровни шкалы (по возрастанию)" : "Варианты ответа"),
        list,
        h("button", { class: "link-btn", onclick: () => { opts.push({ label: "", description: "" }); rerender(); } },
          isScore ? "+ добавить уровень" : "+ добавить вариант"),
        isScore
          ? h("div", { class: "field-note" }, "Порядок уровней задаёт шкалу: от низшего к высшему. Ожидание считается по индексам 0…n−1.")
          : null,
      ),
    );
  }
  return card;
}
