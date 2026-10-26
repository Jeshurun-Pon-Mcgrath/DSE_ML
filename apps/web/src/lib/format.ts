export const human = (s: string) =>
  (s === s.toUpperCase() ? s.toLowerCase() : s)
    .replaceAll('_', ' ')
    .replace(/\b\w/g, (x) => x.toUpperCase());
export const fmt = (n: number | null | undefined, digits = 2) =>
  n == null
    ? 'Unavailable'
    : new Intl.NumberFormat('en-IN', { maximumFractionDigits: digits }).format(n);
export const pct = (n: number | null | undefined, digits = 2) =>
  n == null ? 'Unavailable' : `${fmt(n * 100, digits)}%`;
export const amount = (n: number | null | undefined) =>
  n == null
    ? 'Unavailable'
    : new Intl.NumberFormat('en-IN', {
        style: 'currency',
        currency: 'INR',
        maximumFractionDigits: 2,
      }).format(n);
export const elapsed = (n: number) =>
  `Day ${Math.floor(n / 86400)} · ${Math.floor((n % 86400) / 3600)}h ${Math.floor((n % 3600) / 60)}m`;
