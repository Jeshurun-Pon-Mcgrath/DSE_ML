import { describe, expect, it } from 'vitest';
import { transactionSchema, type Transaction } from '@/lib/api';
import { buildTransactionNetwork, relatedNodeIds } from './graph';

function transaction(
  id: string,
  relationships: { entity?: string | null; merchant?: string | null; device?: string | null },
): Transaction {
  return transactionSchema.parse({
    transaction: {
      TransactionID: id,
      TransactionAmt: 100,
      TransactionDT: 120,
      behavioural_entity_id: relationships.entity ?? null,
      merchant_id: relationships.merchant ?? null,
      device_id: relationships.device ?? null,
    },
    prediction: {
      transaction_id: id,
      risk: { score: 0.4, level: 'ELEVATED', investigation_required: true },
      supervised: { probability: 0.3, prediction: false, model_version: 'fixture-only' },
      anomaly: { score: 0.9, flag: false, diagnostic_only: true, abrf_weight: 0 },
      behaviour: { score: 0.5, confidence: 0.7, history_count: 5, available: true },
      abrf: {
        version: 'abrf-v1',
        base_weights: { wp: 0.9, wa: 0, wb: 0.1 },
        effective_weights: { supervised: 0.93, anomaly: 0, behaviour: 0.07 },
        contributions: { supervised: 0.28, anomaly: 0, behaviour: 0.04 },
        reason_codes: [],
      },
      explanation: {
        status: 'NOT_REQUESTED',
        top_increasing_features: [],
        top_decreasing_features: [],
      },
      versions: {},
    },
    investigation: 'UNREVIEWED',
  });
}

describe('transaction intelligence graph', () => {
  it('creates only relationships present in source records and deduplicates shared nodes', () => {
    const network = buildTransactionNetwork([
      transaction('tx-1', { entity: 'entity-1', merchant: 'merchant-1', device: null }),
      transaction('tx-2', { entity: 'entity-1', merchant: '', device: 'device-2' }),
    ]);
    expect(network.transactionCount).toBe(2);
    expect(network.edges).toHaveLength(4);
    expect(network.nodes.filter((node) => node.kind === 'entity')).toHaveLength(1);
    expect(network.nodes.some((node) => node.id.startsWith('device:null'))).toBe(false);
    expect(network.nodes.some((node) => node.id.startsWith('merchant:'))).toBe(true);
  });

  it('returns exactly the selected transaction and its observed context', () => {
    const network = buildTransactionNetwork([
      transaction('tx-1', { entity: 'entity-1', merchant: 'merchant-1', device: 'device-1' }),
      transaction('tx-2', { entity: 'entity-2', merchant: 'merchant-2', device: null }),
    ]);
    expect([...relatedNodeIds(network, 'tx-1')].sort()).toEqual(
      ['device:device-1', 'entity:entity-1', 'merchant:merchant-1', 'transaction:tx-1'].sort(),
    );
  });

  it('is deterministic and enforces the requested node limit', () => {
    const records = [
      transaction('tx-1', { entity: 'entity-1' }),
      transaction('tx-2', { entity: 'entity-2' }),
    ];
    const first = buildTransactionNetwork(records, 1);
    const second = buildTransactionNetwork(records, 1);
    expect(first).toEqual(second);
    expect(first.transactionCount).toBe(1);
    expect(first.limited).toBe(true);
  });
});
