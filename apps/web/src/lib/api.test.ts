import { afterEach, describe, expect, it, vi } from 'vitest';
import { z } from 'zod';
import { ApiError, number, object, predictionSchema, request, valueAt } from './api';
import { elapsed, fmt, pct } from './format';
// Synthetic UI contract fixtures only. No model fitting, scoring or report generation.
const fixture = {
  transaction_id: 'ui-test',
  risk: { score: 0.4, level: 'ELEVATED', investigation_required: true },
  supervised: { probability: 0.3, prediction: false, model_version: 'test-only' },
  anomaly: { score: 0.9, flag: true, diagnostic_only: true, abrf_weight: 0 },
  behaviour: { score: null, confidence: 0, history_count: 0, available: false },
  abrf: {
    version: 'test-only',
    base_weights: { wp: 0.9, wa: 0, wb: 0.1 },
    effective_weights: { supervised: 1, anomaly: 0, behaviour: 0 },
    contributions: { supervised: 0.3, anomaly: 0, behaviour: 0 },
    reason_codes: [],
  },
  explanation: {
    status: 'NOT_REQUESTED',
    top_increasing_features: [],
    top_decreasing_features: [],
  },
  versions: {},
};
afterEach(() => vi.unstubAllGlobals());
describe('Typed API boundaries', () => {
  it('accepts explicit cold-start unavailable evidence', () =>
    expect(predictionSchema.parse(fixture).behaviour.score).toBeNull());
  it('rejects nonzero anomaly influence', () =>
    expect(
      predictionSchema.safeParse({ ...fixture, anomaly: { ...fixture.anomaly, abrf_weight: 0.1 } })
        .success,
    ).toBe(false));
  it.each([-1, 1.01, NaN, Infinity])('rejects invalid risk score %s', (score) =>
    expect(
      predictionSchema.safeParse({ ...fixture, risk: { ...fixture.risk, score } }).success,
    ).toBe(false),
  );
  it('parses successful responses', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(Response.json({ value: 12 })));
    await expect(request('/test', z.object({ value: z.number() }))).resolves.toEqual({ value: 12 });
  });
  it('uses structured errors without raw stack traces', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValue(
          Response.json(
            { error: { message: 'Transaction not found', request_id: 'test-request' } },
            { status: 404 },
          ),
        ),
    );
    await expect(request('/test', z.json())).rejects.toMatchObject({
      status: 404,
      requestId: 'test-request',
      message: 'Transaction not found',
    });
  });
  it('rejects incompatible success contracts', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(Response.json({ value: 'wrong' })));
    await expect(request('/test', z.object({ value: z.number() }))).rejects.toBeInstanceOf(
      ApiError,
    );
  });
  it('reports network failures safely', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('internal network stack')));
    const result = request('/test', z.json());
    await expect(result).rejects.toMatchObject({ status: 0 });
    await expect(result).rejects.toThrow('The API is unavailable.');
  });
});
describe('Display truthfulness', () => {
  it('preserves unknown values', () => {
    expect(fmt(null)).toBe('Unavailable');
    expect(pct(undefined)).toBe('Unavailable');
  });
  it('formats elapsed time without a calendar', () => expect(elapsed(90000)).toBe('Day 1 · 1h 0m'));
  it('formats API proportions as percentages', () => expect(pct(0.1234)).toBe('12.34%'));
  it('handles bounded research JSON safely', () => {
    expect(number(Infinity)).toBeNull();
    expect(object(null)).toEqual({});
    expect(valueAt({ a: { b: 2 } }, 'a', 'b')).toBe(2);
  });
});
