import { z } from 'zod';
import { env } from '@/config/env';

const score = z.number().finite().min(0).max(1);
export const levels = ['LOW', 'GUARDED', 'ELEVATED', 'HIGH', 'CRITICAL'] as const;
export const states = ['UNREVIEWED', 'UNDER_REVIEW', 'CLEARED', 'ESCALATED'] as const;
export const riskSchema = z.object({
  score,
  level: z.enum(levels),
  investigation_required: z.boolean(),
});
export const featureSchema = z.object({
  feature: z.string(),
  display_name: z.string(),
  feature_value: z.number(),
  shap_value: z.number(),
  interpretability: z.string(),
  feature_family: z.string(),
  statement: z.string(),
  source_columns: z.array(z.string()).optional(),
  raw_or_derived: z.string().optional(),
  manifest_interpretability: z.string().optional(),
});
const weights = z.record(z.string(), z.number());
export const predictionSchema = z.object({
  transaction_id: z.string(),
  risk: riskSchema,
  supervised: z.object({ probability: score, prediction: z.boolean(), model_version: z.string() }),
  anomaly: z.object({
    score,
    flag: z.boolean(),
    diagnostic_only: z.literal(true),
    abrf_weight: z.literal(0),
  }),
  behaviour: z.object({
    score: score.nullable(),
    confidence: score,
    history_count: z.number(),
    available: z.boolean(),
  }),
  abrf: z.object({
    version: z.string(),
    base_weights: weights,
    effective_weights: weights,
    contributions: weights,
    reason_codes: z.array(z.string()),
  }),
  explanation: z.object({
    status: z.string(),
    top_increasing_features: z.array(featureSchema),
    top_decreasing_features: z.array(featureSchema),
  }),
  versions: z.record(z.string(), z.string()),
});
export const transactionSchema = z.object({
  transaction: z.record(z.string(), z.union([z.string(), z.number(), z.null()])),
  prediction: predictionSchema,
  investigation: z.enum(states),
});
export const transactionPageSchema = z.object({
  total: z.number(),
  page: z.number(),
  page_size: z.number(),
  items: z.array(transactionSchema),
});
export const metricsSchema = z.object({
  precision: score,
  recall: score,
  f1: score,
  pr_auc: score.nullable(),
  roc_auc: score.nullable(),
  fpr: score,
  fnr: score,
  confusion: z.object({ tp: z.number(), fp: z.number(), tn: z.number(), fn: z.number() }),
  hard_negatives: z.object({ count: z.number(), flagged: z.number(), rate: score }),
});
export const finalSchema = z.object({
  transaction_count: z.number(),
  fraud_count: z.number(),
  fraud_prevalence: score,
  hard_negative_count: z.number(),
  holdout_fingerprint: z.string(),
  supervised: metricsSchema,
  abrf: metricsSchema,
  improvement: z.record(z.string(), z.json()),
});
export const rateSchema = z.object({
  count: z.number(),
  flagged: z.number(),
  rate: score.nullable(),
});
export const scenariosSchema = z.record(
  z.string(),
  z.object({ count: z.number(), supervised: rateSchema, abrf: rateSchema, anomaly: rateSchema }),
);
export const hardSchema = z.record(
  z.string(),
  z.object({ count: z.number(), supervised: rateSchema, abrf: rateSchema }),
);
export const dashboardSchema = z.object({
  transaction_count: z.number(),
  investigation_count: z.number(),
  average_risk_score: score.nullable(),
  risk_level_distribution: z.record(z.string(), z.number()),
  risk_trend: z.array(
    z.object({
      elapsed_day_index: z.number(),
      transactions: z.number(),
      average_risk_score: score,
    }),
  ),
  recent_transaction_activity: z.array(transactionSchema),
  recent_high_risk_transactions: z.array(transactionSchema),
  scope: z.string(),
});
export const modelSchema = z.object({
  dataset_type: z.string(),
  notice: z.string(),
  supervised_model: z.string(),
  supervised_model_id: z.string(),
  supervised_calibration: z.string(),
  development_fingerprint: z.string(),
  final_holdout_fingerprint: z.string(),
  model_version: z.string(),
  feature_version: z.string(),
  anomaly_version: z.string(),
  behaviour_version: z.string(),
  abrf_version: z.string(),
  explainability_version: z.string(),
  shap_version: z.string(),
  system_status: z.string(),
  minimum_elapsed_seconds: z.number(),
  final_holdout_status: z.string(),
});
const componentSchema = z.object({
  signal: z.string(),
  source: z.string(),
  score: score.nullable(),
  raw_value: z.number().nullable(),
  contribution: score.nullable(),
  available: z.boolean(),
  window: z.string().nullable(),
});
export const explanationSchema = z.object({
  transaction_id: z.string(),
  version: z.string(),
  risk_summary: z.object({
    final_risk_score: score,
    risk_level: z.enum(levels),
    investigation_flag: z.boolean(),
  }),
  supervised: z.object({
    model_version: z.string(),
    explained_model: z.literal('underlying_decision_tree'),
    raw_probability: score,
    calibrated_probability: score,
    threshold: score,
    shap_status: z.string(),
    base_value: z.number().nullable(),
    top_positive_features: z.array(featureSchema),
    top_negative_features: z.array(featureSchema),
  }),
  behaviour: z.object({
    behaviour_score: score.nullable(),
    behaviour_confidence: score,
    history_count: z.number(),
    history_available: z.boolean(),
    components: z.array(componentSchema),
    confidence_explanation: z.object({ supporting_measurements: z.array(z.string()) }),
  }),
  anomaly: z.object({
    score,
    threshold: score,
    flag: z.boolean(),
    diagnostic_only: z.literal(true),
    effective_abrf_weight: z.literal(0),
    influenced_final_risk: z.literal(false),
  }),
  fusion: z.object({
    version: z.string(),
    supervised_probability: score,
    anomaly_score: score,
    behaviour_score: score.nullable(),
    behaviour_confidence: score,
    base_weights: weights,
    effective_weights: weights,
    weighted_contributions: weights,
    final_risk_score: score,
    risk_level: z.enum(levels),
    decision_threshold: score,
    investigation_flag: z.boolean(),
    reason_codes: z.array(z.string()),
  }),
  formula_trace: z.object({
    formula: z.string(),
    confidence_rule: z.string(),
    transformed_confidence: score,
    denominator: z.number(),
    unnormalized_contributions: weights,
  }),
  limitations: z.array(z.string()),
});
export const auditSchema = z.object({
  total: z.number(),
  page: z.number(),
  page_size: z.number(),
  items: z.array(
    z.object({
      id: z.number(),
      transaction_id: z.string(),
      action: z.string(),
      timestamp: z.string(),
      metadata: z.record(z.string(), z.json()),
    }),
  ),
});
export const healthSchema = z.object({
  api_ready: z.boolean(),
  inference_ready: z.boolean(),
  supervised_loaded: z.boolean(),
  anomaly_loaded: z.boolean(),
  behaviour_loaded: z.boolean(),
  abrf_loaded: z.boolean(),
  shap_ready: z.boolean(),
  database_ready: z.boolean(),
  unavailable_reason: z.string().nullable(),
});
export type Transaction = z.infer<typeof transactionSchema>;
export type Prediction = z.infer<typeof predictionSchema>;
export type Explanation = z.infer<typeof explanationSchema>;
export type FinalEvaluation = z.infer<typeof finalSchema>;
export type Metric = z.infer<typeof metricsSchema>;
export type Feature = z.infer<typeof featureSchema>;
export type Json = z.infer<ReturnType<typeof z.json>>;
export type ReviewState = (typeof states)[number];
export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly requestId?: string,
  ) {
    super(message);
  }
}
export async function request<T>(
  path: string,
  schema: z.ZodType<T>,
  options: RequestInit = {},
  signal?: AbortSignal,
): Promise<T> {
  const origin = import.meta.env.DEV ? '' : env.VITE_API_BASE_URL;
  let response: Response;
  try {
    response = await fetch(`${origin}/api/v1${path}`, {
      ...options,
      headers: { 'Content-Type': 'application/json', ...options.headers },
      signal: signal
        ? AbortSignal.any([signal, AbortSignal.timeout(60000)])
        : AbortSignal.timeout(60000),
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error;
    throw new ApiError('The API is unavailable. Check the backend connection and try again.', 0);
  }
  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const parsed = z
      .object({ error: z.object({ message: z.string(), request_id: z.string().optional() }) })
      .safeParse(payload);
    throw new ApiError(
      parsed.success ? parsed.data.error.message : 'The request could not be completed.',
      response.status,
      parsed.success ? parsed.data.error.request_id : undefined,
    );
  }
  const result = schema.safeParse(payload);
  if (!result.success)
    throw new ApiError(
      'The API returned an incompatible response. Refresh or contact the operator.',
      502,
    );
  return result.data;
}
export function object(value: Json | undefined): Record<string, Json> {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? value : {};
}
export function list(value: Json | undefined): Json[] {
  return Array.isArray(value) ? value : [];
}
export function valueAt(value: Json | undefined, ...keys: string[]): Json | undefined {
  return keys.reduce<Json | undefined>((v, k) => object(v)[k], value);
}
export function number(value: Json | undefined): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}
export function text(value: Json | undefined): string {
  return typeof value === 'string' ? value : 'Unavailable';
}
