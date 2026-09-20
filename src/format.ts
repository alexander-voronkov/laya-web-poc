export const mb = (n: number) => (n / 1e6).toFixed(1);

export const pct = (p: number, digits = 1) => `${(p * 100).toFixed(digits)}%`;

/** Sub-10ms readings are meaningful here (the head pass is ~2ms) and rounding them
 *  to 0 would read as "free". */
export const ms = (n: number) => (n < 10 ? `${n.toFixed(1)} ms` : `${n.toFixed(0)} ms`);

export const sec = (n: number) => `${(n / 1000).toFixed(n < 10_000 ? 2 : 1)} s`;

