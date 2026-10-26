import type { Transaction } from '@/lib/api';

export type NetworkNodeKind = 'transaction' | 'entity' | 'merchant' | 'device';
export type Point3 = readonly [number, number, number];

export interface TransactionNetworkNode {
  id: string;
  kind: NetworkNodeKind;
  label: string;
  position: Point3;
  transaction?: Transaction;
}

export interface TransactionNetworkEdge {
  id: string;
  source: string;
  target: string;
  relationship: Exclude<NetworkNodeKind, 'transaction'>;
}

export interface TransactionNetwork {
  nodes: TransactionNetworkNode[];
  edges: TransactionNetworkEdge[];
  transactionCount: number;
  limited: boolean;
}

export const TRANSACTION_NETWORK_LIMIT = 24;

function textValue(value: string | number | null | undefined): string | null {
  if (value === null || value === undefined) return null;
  const valueText = String(value).trim();
  return valueText ? valueText : null;
}

function stableUnit(value: string): number {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0) / 4294967295;
}

function relationshipPosition(kind: Exclude<NetworkNodeKind, 'transaction'>, key: string): Point3 {
  const angle = stableUnit(`${kind}:${key}`) * Math.PI * 2;
  const radius = kind === 'entity' ? 4.7 : kind === 'merchant' ? 5.8 : 6.8;
  const y = kind === 'entity' ? -0.6 : kind === 'merchant' ? 0.15 : 0.9;
  return [Math.cos(angle) * radius, y, Math.sin(angle) * radius];
}

function transactionPosition(transaction: Transaction, index: number, total: number): Point3 {
  const angle = (index / Math.max(total, 1)) * Math.PI * 2 - Math.PI / 2;
  const risk = transaction.prediction.risk.score;
  const radius = 2.4 + stableUnit(transaction.prediction.transaction_id) * 0.8;
  return [Math.cos(angle) * radius, -0.25 + risk * 1.8, Math.sin(angle) * radius];
}

function nodeId(kind: NetworkNodeKind, key: string): string {
  return `${kind}:${key}`;
}

export function buildTransactionNetwork(
  transactions: Transaction[],
  limit = TRANSACTION_NETWORK_LIMIT,
): TransactionNetwork {
  const bounded = transactions.slice(0, Math.max(0, limit));
  const nodes = new Map<string, TransactionNetworkNode>();
  const edges: TransactionNetworkEdge[] = [];

  bounded.forEach((transaction, index) => {
    const transactionKey = transaction.prediction.transaction_id;
    const transactionNodeId = nodeId('transaction', transactionKey);
    nodes.set(transactionNodeId, {
      id: transactionNodeId,
      kind: 'transaction',
      label: transactionKey,
      position: transactionPosition(transaction, index, bounded.length),
      transaction,
    });

    const relationships: Array<
      [Exclude<NetworkNodeKind, 'transaction'>, string | number | null | undefined]
    > = [
      ['entity', transaction.transaction.behavioural_entity_id],
      ['merchant', transaction.transaction.merchant_id],
      ['device', transaction.transaction.device_id],
    ];

    relationships.forEach(([kind, rawValue]) => {
      const value = textValue(rawValue);
      if (!value) return;
      const relatedNodeId = nodeId(kind, value);
      if (!nodes.has(relatedNodeId)) {
        nodes.set(relatedNodeId, {
          id: relatedNodeId,
          kind,
          label: value,
          position: relationshipPosition(kind, value),
        });
      }
      edges.push({
        id: `${transactionNodeId}->${relatedNodeId}`,
        source: transactionNodeId,
        target: relatedNodeId,
        relationship: kind,
      });
    });
  });

  return {
    nodes: [...nodes.values()],
    edges,
    transactionCount: bounded.length,
    limited: transactions.length > bounded.length,
  };
}

export function relatedNodeIds(
  network: TransactionNetwork,
  transactionId: string | null,
): Set<string> {
  if (!transactionId) return new Set();
  const selected = nodeId('transaction', transactionId);
  const related = new Set<string>([selected]);
  network.edges.forEach((edge) => {
    if (edge.source === selected) related.add(edge.target);
  });
  return related;
}

export function networkNodeForTransaction(
  network: TransactionNetwork,
  transactionId: string,
): TransactionNetworkNode | undefined {
  return network.nodes.find((node) => node.id === nodeId('transaction', transactionId));
}
