import { useState } from "react";
import { TYPE_LABELS, scoredItems, type QuestionItem } from "../questions";
import type { Answer, QuestionTelemetry } from "../laya/types";
import { ms, pct } from "../format";

interface Props {
  question: QuestionItem;
  index: number;
  answer: Answer;
  telemetry: QuestionTelemetry;
}

export function AnswerCard({ question: q, index, answer: ans, telemetry }: Props) {
  const [open, setOpen] = useState(false);
  // Clipboard access is refused outside a secure context and can be denied by policy.
  // Failing silently means the user pastes whatever was in the buffer before -- very
  // plausibly the previous answer card -- into a report.
  const [copied, setCopied] = useState<"ok" | "fail" | null>(null);
  const copy = () => {
    const payload = { id: q.id, type: q.type, question: q.text, answer: ans, telemetry };
    const text = JSON.stringify(payload, null, 2);
    const done = (r: "ok" | "fail") => { setCopied(r); setTimeout(() => setCopied(null), 2000); };
    if (!navigator.clipboard) { done("fail"); return; }
    navigator.clipboard.writeText(text).then(() => done("ok"), () => done("fail"));
  };

  return (
    <div className={`ans-card type-${ans.type}`}>
      <div className="ans-head">
        <span className="ans-num">Question {index + 1}</span>
        <span className="ans-type">{TYPE_LABELS[q.type]}</span>
        <span className="ans-ms" title={telemetry.batchSize > 1 ? `one forward pass shared by ${telemetry.batchSize} questions` : undefined}>
          {ms(telemetry.totalMs)}{telemetry.batchSize > 1 ? ` / ${telemetry.batchSize}` : ""}
        </span>
        <button className="icon-btn" title="Copy the answer with its metrics" onClick={copy}>⧉</button>
        {copied && (
          <span className={copied === "ok" ? "muted" : "error"}>
            {copied === "ok" ? "copied" : "clipboard unavailable"}
          </span>
        )}
      </div>
      <div className="ans-text">{q.text || TYPE_LABELS[q.type]}</div>

      {ans.type === "noul" && <NoulBody p={ans.noul} />}
      {ans.type === "choice" && <ChoiceBody q={q} probabilities={ans.probabilities} confidence={ans.confidence} />}
      {ans.type === "score" && (
        <ScoreBody q={q} probabilities={ans.probabilities} score={ans.score} confidence={ans.confidence} />
      )}

      <button className="link-btn" onClick={() => setOpen(!open)}>
        {open ? "hide breakdown" : "breakdown: tokens, timing, temperature"}
      </button>
      {open && <Breakdown t={telemetry} act={ans.rl_agent.act_probability} />}
    </div>
  );
}

function NoulBody({ p }: { p: number }) {
  const v = Math.min(100, Math.max(0, p * 100));
  return (
    <>
      <div className="noul-big">{v.toFixed(1)}%</div>
      <div className="bar"><div className="bar-fill" style={{ width: `${v.toFixed(1)}%` }} /></div>
      <div className="ans-sub">probability of yes (true) · no: {(100 - v).toFixed(1)}%</div>
    </>
  );
}

function ChoiceBody({ q, probabilities, confidence }: {
  q: QuestionItem; probabilities: Record<string, number>; confidence: number;
}) {
  // scoredItems, not a local filter: this is the same list the request was built from,
  // so the labels here line up with the keys the model answered under. A local filter
  // is what previously let the display and the request disagree.
  const entries = scoredItems(q).map((o) => ({
    label: o.label,
    // Missing means "the model never scored this" -- a label renamed after the run,
    // say. Drawing it as 0.0% would read as a confident zero.
    p: o.label in probabilities ? probabilities[o.label] : null,
  }));
  const maxP = Math.max(...entries.map((e) => e.p ?? -1));
  // First index, not "every entry equal to the max": a tie would otherwise highlight
  // several rows and read as several winners.
  const topIndex = entries.findIndex((e) => e.p !== null && e.p === maxP);
  return (
    <>
      <div className="dist">
        {entries.map((e, i) => <DistRow key={e.label} label={e.label} p={e.p} top={i === topIndex} />)}
      </div>
      <div className="ans-sub">
        top: {entries[topIndex]?.label ?? "—"} · confidence {pct(confidence, 0)}
      </div>
    </>
  );
}

function ScoreBody({ q, probabilities, score, confidence }: {
  q: QuestionItem; probabilities: Record<string, number>; score: number; confidence: number;
}) {
  // Same list the request was built from. The indices the model answers under are
  // positions in *that* list, so filtering here independently is what used to put each
  // label next to its neighbour's probability.
  const levels = scoredItems(q).map((l) => l.label);
  const probs = levels.map((_, i) => (String(i) in probabilities ? probabilities[String(i)] : null));
  const maxP = Math.max(...probs.map((p) => p ?? -1));
  const topIndex = probs.findIndex((p) => p !== null && p === maxP);
  const pos = levels.length > 1 ? (score / (levels.length - 1)) * 100 : 50;
  return (
    <>
      <div className="dist">
        {levels.map((l, i) => <DistRow key={l} label={l} p={probs[i]} top={i === topIndex} />)}
      </div>
      <div className="scale-wrap">
        <div className="scale-track">
          <div
            className="scale-marker"
            style={{ left: `${Math.min(100, Math.max(0, pos)).toFixed(1)}%` }}
            title={`expected ${score.toFixed(2)}`}
          />
        </div>
        <div className="scale-ticks">{levels.map((_, i) => <span key={i}>{i}</span>)}</div>
      </div>
      <div className="ans-sub">
        expected {score.toFixed(2)} of {levels.length - 1} · confidence {pct(confidence, 0)}
      </div>
    </>
  );
}

/** `p === null` means the model returned no probability under this label, which is a
 *  different statement from "it returned zero" and is rendered as one. */
function DistRow({ label, p, top }: { label: string; p: number | null; top: boolean }) {
  return (
    <div className={`dist-row${top ? " top" : ""}`}>
      <span className="dist-lbl">{label}</span>
      <div className="dist-bar">
        <div className="dist-fill" style={{ width: p === null ? "0%" : `${(p * 100).toFixed(1)}%` }} />
      </div>
      <span className={p === null ? "dist-pct muted" : "dist-pct"}>{p === null ? "no answer" : pct(p)}</span>
    </div>
  );
}

function Breakdown({ t, act }: { t: QuestionTelemetry; act: number }) {
  const s = t.stats;
  const rows: [string, string][] = [
    ["Sequence tokens", `${s.totalTokens} = question ${s.headTokens} + options ${s.optionTokens} + text ${s.stateTokensUsed} + 3 special`],
    ["Text", s.stateTokensUsed < s.stateTokens
      ? `${s.stateTokensUsed} of ${s.stateTokens} tokens — tail dropped`
      : `${s.stateTokens} tokens, in full`],
    ["Wording", s.headTokens < s.headTokensFull
      ? `${s.headTokens} of ${s.headTokensFull} tokens — the end was cut`
      : `${s.headTokensFull} tokens, in full`],
    ["Options", s.optionsShrunk
      ? `${s.optionTokens} of ${s.optionTokensFull} tokens — each option shortened individually`
      : `${s.optionTokens} tokens, in full`],
    ["Timing", t.batchSize > 1
      ? `encoder ${ms(t.encoderMs)} · head ${ms(t.headMs)} — one forward pass shared by ${t.batchSize} questions · sequence built in ${ms(t.buildMs)}`
      : `total ${ms(t.totalMs)} · encoder ${ms(t.encoderMs)} · head ${ms(t.headMs)} · build ${ms(t.buildMs)}`],
    ["Encoder throughput", t.batchSize > 1
      ? "measured per batch, see the metrics panel"
      : `${(s.totalTokens / (t.encoderMs / 1000)).toFixed(0)} tokens/s`],
    ["Temperature", `${t.temperature.toFixed(4)} (bucket ${t.temperatureBucket}, ${t.options} options)`],
    // Documented on the model card as saturated at 1.000 on every input tested, so
    // showing it without this line would invite reading signal into a constant.
    ["act_probability", `${act.toFixed(3)} — the act head is saturated on this checkpoint and carries no signal`],
  ];
  return (
    <div className="breakdown">
      {rows.map(([k, v]) => (
        <div className="m-row" key={k}><span className="m-k">{k}</span><span className="m-v">{v}</span></div>
      ))}
    </div>
  );
}
