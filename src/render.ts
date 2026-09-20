import { h } from "./dom";
import { TYPE_LABELS, type QuestionItem } from "./state";
import type { Answer } from "./laya/types";

export function answerCard(q: QuestionItem, index: number, ans: Answer, ms: number): HTMLElement {
  const card = h("div", { class: `ans-card type-${ans.type}`, "data-ans": q.id });
  card.append(
    h("div", { class: "ans-head" },
      h("span", { class: "ans-num" }, `Вопрос ${index + 1}`),
      h("span", { class: "ans-type" }, TYPE_LABELS[q.type]),
      h("span", { class: "ans-ms" }, `${ms < 10 ? ms.toFixed(1) : ms.toFixed(0)} мс`),
      h("button", { class: "icon-btn", title: "Скопировать ответ", onclick: (e: Event) => copyAnswer(e, q, ans) }, "⧉"),
    ),
    h("div", { class: "ans-text" }, q.text || TYPE_LABELS[q.type]),
  );

  if (ans.type === "noul") {
    const pct = ans.noul * 100;
    card.append(
      h("div", { class: "noul-big" }, `${pct.toFixed(1)}%`),
      h("div", { class: "bar" }, h("div", { class: "bar-fill", style: `width:${Math.min(100, Math.max(0, pct)).toFixed(1)}%` })),
      h("div", { class: "ans-sub" }, `вероятность «да» (true) · «нет»: ${(100 - pct).toFixed(1)}%`),
    );
    return card;
  }

  if (ans.type === "choice") {
    const entries = q.options
      .filter((o) => o.label.trim())
      .map((o) => ({ label: o.label.trim(), p: ans.probabilities[o.label.trim()] ?? 0 }));
    let top = entries[0];
    for (const e of entries) if (e.p > top.p) top = e;
    card.append(h("div", { class: "dist" }, ...entries.map((e) => distRow(e.label, e.p, e === top))));
    card.append(h("div", { class: "ans-sub" }, `топ: ${top.label} · уверенность ${(ans.confidence * 100).toFixed(0)}%`));
    return card;
  }

  // score: distribution over ordered levels + expected value marker
  const levels = q.levels.filter((l) => l.label.trim());
  const probs = levels.map((_, idx) => ans.probabilities[String(idx)] ?? 0);
  const maxP = Math.max(...probs);
  const pos = levels.length > 1 ? (ans.score / (levels.length - 1)) * 100 : 50;
  card.append(
    h("div", { class: "dist" }, ...levels.map((l, idx) => distRow(l.label.trim(), probs[idx], probs[idx] === maxP))),
    h("div", { class: "scale-wrap" },
      h("div", { class: "scale-track" },
        h("div", { class: "scale-marker", style: `left:${Math.min(100, Math.max(0, pos)).toFixed(1)}%`, title: `ожидание ${ans.score.toFixed(2)}` }),
      ),
      h("div", { class: "scale-ticks" }, ...levels.map((_, idx) => h("span", null, String(idx)))),
    ),
    h("div", { class: "ans-sub" },
      `ожидание ${ans.score.toFixed(2)} из ${levels.length - 1} · уверенность ${(ans.confidence * 100).toFixed(0)}%`),
  );
  return card;
}

function distRow(label: string, p: number, top: boolean): HTMLElement {
  return h("div", { class: `dist-row${top ? " top" : ""}` },
    h("span", { class: "dist-lbl" }, label),
    h("div", { class: "dist-bar" }, h("div", { class: "dist-fill", style: `width:${(p * 100).toFixed(1)}%` })),
    h("span", { class: "dist-pct" }, `${(p * 100).toFixed(1)}%`),
  );
}

function copyAnswer(_e: Event, q: QuestionItem, ans: Answer): void {
  const payload = { id: q.id, type: q.type, question: q.text, answer: ans };
  navigator.clipboard?.writeText(JSON.stringify(payload, null, 2)).catch(() => {});
}
