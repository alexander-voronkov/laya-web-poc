export const mb = (n: number) => (n / 1e6).toFixed(1);

export const pct = (p: number, digits = 1) => `${(p * 100).toFixed(digits)}%`;

/** Sub-10ms readings are meaningful here (the head pass is ~2ms) and rounding them
 *  to 0 would read as "free". */
export const ms = (n: number) => (n < 10 ? `${n.toFixed(1)} мс` : `${n.toFixed(0)} мс`);

export const sec = (n: number) => `${(n / 1000).toFixed(n < 10_000 ? 2 : 1)} с`;

export function plural(n: number, one: string, few: string, many: string): string {
  const m10 = n % 10;
  const m100 = n % 100;
  if (m10 === 1 && m100 !== 11) return one;
  if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return few;
  return many;
}
