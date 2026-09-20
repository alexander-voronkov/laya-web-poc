export const mb = (n: number) => (n / 1e6).toFixed(1);

export const pct = (p: number, digits = 1) => `${(p * 100).toFixed(digits)}%`;

/** Sub-10ms readings are meaningful here (the head pass can be ~2ms) and rounding them
 *  to 0 would read as "free". Past ten seconds the opposite applies: a forward pass on
 *  a slow two-core box really does take 45 s, and "45569 ms" is a number nobody reads
 *  as three quarters of a minute. */
export const ms = (n: number) =>
  n < 10 ? `${n.toFixed(1)} ms` : n < 10_000 ? `${n.toFixed(0)} ms` : `${(n / 1000).toFixed(1)} s`;

export const sec = (n: number) => `${(n / 1000).toFixed(n < 10_000 ? 2 : 1)} s`;

