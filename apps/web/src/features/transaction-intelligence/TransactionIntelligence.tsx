import {
  Component,
  lazy,
  Suspense,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import { Box, BoxSelect, RotateCcw } from 'lucide-react';
import { Link } from 'react-router-dom';
import { Button, DefinitionList, EmptyState, Panel, RiskBadge, Skeleton } from '@/components/ui';
import type { Transaction } from '@/lib/api';
import { fmt, pct } from '@/lib/format';
import { buildTransactionNetwork, relatedNodeIds } from './graph';

const TransactionNetworkScene = lazy(() => import('./TransactionNetworkScene'));

type ViewMode = 'three' | 'fallback';

interface BoundaryProps {
  children: ReactNode;
  onFailure: () => void;
}

class SceneBoundary extends Component<BoundaryProps, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  componentDidCatch() {
    this.props.onFailure();
  }
  render() {
    return this.state.failed ? null : this.props.children;
  }
}

function hasWebGL(): boolean {
  try {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('webgl2') ?? canvas.getContext('webgl');
    context?.getExtension('WEBGL_lose_context')?.loseContext();
    return Boolean(context);
  } catch {
    return false;
  }
}

function initialFallbackReason(webglAvailable: boolean): string | null {
  if (!webglAvailable) return 'WebGL is unavailable in this browser.';
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches)
    return 'Reduced motion is enabled.';
  if (window.matchMedia('(max-width: 767px)').matches)
    return 'The compact 2D view is used on smaller screens.';
  const device = navigator as Navigator & { deviceMemory?: number };
  if ((device.deviceMemory ?? 8) <= 2 || navigator.hardwareConcurrency <= 2)
    return 'The compact view is used on this lower-resource device.';
  return null;
}

function initialDisplayState(): {
  viewMode: ViewMode;
  fallbackReason: string | null;
  webglAvailable: boolean;
} {
  const webglAvailable = hasWebGL();
  if (!webglAvailable)
    return {
      viewMode: 'fallback',
      fallbackReason: 'WebGL is unavailable in this browser.',
      webglAvailable,
    };
  const fallbackReason = initialFallbackReason(webglAvailable);
  return { viewMode: fallbackReason ? 'fallback' : 'three', fallbackReason, webglAvailable };
}

function NetworkFallback({
  transactions,
  selectedId,
  onSelect,
}: {
  transactions: Transaction[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const selectedIndex = transactions.findIndex(
    (transaction) => transaction.prediction.transaction_id === selectedId,
  );
  return (
    <div className="network-fallback" data-testid="transaction-network-fallback">
      <svg viewBox="0 0 720 270" role="img" aria-labelledby="network-2d-title network-2d-desc">
        <title id="network-2d-title">Transaction relationship network</title>
        <desc id="network-2d-desc">
          Transactions connect to their observed behavioural entity, merchant and device context.
          Node size and ring count indicate fused risk.
        </desc>
        {transactions.map((transaction, index) => {
          const x = 48 + (index % 6) * 122;
          const y = 65 + Math.floor(index / 6) * 100;
          const risk = transaction.prediction.risk;
          const selected = index === selectedIndex;
          const dimmed = selectedIndex >= 0 && !selected;
          const relations = [
            transaction.transaction.behavioural_entity_id,
            transaction.transaction.merchant_id,
            transaction.transaction.device_id,
          ].filter((value) => value !== null && value !== undefined && String(value).trim());
          return (
            <g
              key={transaction.prediction.transaction_id}
              role="button"
              tabIndex={0}
              aria-label={`Select transaction ${transaction.prediction.transaction_id}, ${risk.level} risk`}
              className={`network-record ${dimmed ? 'dimmed' : ''}`}
              onClick={() => onSelect(transaction.prediction.transaction_id)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault();
                  onSelect(transaction.prediction.transaction_id);
                }
              }}
            >
              {relations.map((_, relationIndex) => {
                const angle = relationIndex * 2.1 - 1;
                const relationX = x + Math.cos(angle) * 32;
                const relationY = y + Math.sin(angle) * 32;
                return (
                  <g key={relationIndex}>
                    <line x1={x} y1={y} x2={relationX} y2={relationY} className="network-line" />
                    <rect
                      x={relationX - 3.5}
                      y={relationY - 3.5}
                      width="7"
                      height="7"
                      className="network-relation"
                    />
                  </g>
                );
              })}
              <circle
                cx={x}
                cy={y}
                r={9 + risk.score * 8}
                className={`network-node risk-${risk.level.toLowerCase()}`}
              />
              {selected && <circle cx={x} cy={y} r={22} className="network-selection-ring" />}
              <text x={x} y={y + 29} textAnchor="middle">
                {transaction.prediction.transaction_id}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function Legend() {
  return (
    <div className="network-legend" aria-label="Network legend">
      <span>
        <i className="legend-shape transaction" />
        Transaction: size and tone show risk
      </span>
      <span>
        <i className="legend-shape entity" />
        Behavioural entity
      </span>
      <span>
        <i className="legend-shape merchant" />
        Merchant
      </span>
      <span>
        <i className="legend-shape device" />
        Device
      </span>
    </div>
  );
}

export function TransactionIntelligence({ transactions }: { transactions: Transaction[] }) {
  const graphTransactions = useMemo(() => transactions.slice(0, 12), [transactions]);
  const network = useMemo(() => buildTransactionNetwork(graphTransactions), [graphTransactions]);
  const [selectedId, setSelectedId] = useState<string | null>(
    graphTransactions[0]?.prediction.transaction_id ?? null,
  );
  const [display, setDisplay] = useState(initialDisplayState);
  const [resetKey, setResetKey] = useState(0);
  const [visible, setVisible] = useState(() => !('IntersectionObserver' in window));
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const element = containerRef.current;
    if (!element || !('IntersectionObserver' in window)) return undefined;
    const observer = new IntersectionObserver(
      ([entry]) => setVisible(entry?.isIntersecting ?? true),
      {
        rootMargin: '80px',
      },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const onVisibility = () => setVisible(!document.hidden);
    document.addEventListener('visibilitychange', onVisibility);
    return () => document.removeEventListener('visibilitychange', onVisibility);
  }, []);

  if (!transactions.length)
    return (
      <Panel title="3D transaction intelligence">
        <EmptyState
          title="No transaction context"
          description="There are no observed records to visualize."
        />
      </Panel>
    );

  const selected = graphTransactions.find(
    (transaction) => transaction.prediction.transaction_id === selectedId,
  );
  const selectedRelated = relatedNodeIds(network, selectedId).size - (selectedId ? 1 : 0);
  const { viewMode, fallbackReason, webglAvailable } = display;
  const relationshipItems: [string, ReactNode][] = selected
    ? [
        ['Behavioural entity', selected.transaction.behavioural_entity_id],
        ['Merchant', selected.transaction.merchant_id],
        ['Device', selected.transaction.device_id],
      ].flatMap(([label, value]) =>
        value === null || value === undefined || String(value).trim() === ''
          ? []
          : ([[label, String(value)]] as [string, ReactNode][]),
      )
    : [];

  return (
    <Panel
      title="3D transaction intelligence"
      note="A bounded network of recent observed transactions and their recorded relationship context."
      className="transaction-intelligence-panel"
    >
      <div className="network-toolbar">
        <Legend />
        <div className="network-actions">
          {viewMode === 'three' && visible ? (
            <>
              <Button className="secondary" onClick={() => setResetKey((value) => value + 1)}>
                <RotateCcw size={15} /> Reset view
              </Button>
              <Button
                className="secondary"
                onClick={() =>
                  setDisplay((current) => ({
                    ...current,
                    viewMode: 'fallback',
                    fallbackReason: 'The compact 2D view was selected.',
                  }))
                }
              >
                <BoxSelect size={15} /> Use 2D
              </Button>
            </>
          ) : webglAvailable ? (
            <Button
              className="secondary"
              onClick={() => setDisplay((current) => ({ ...current, viewMode: 'three' }))}
            >
              <Box size={15} /> Open 3D
            </Button>
          ) : null}
        </div>
      </div>
      {fallbackReason && viewMode !== 'three' && (
        <p className="network-fallback-note" role="status">
          {fallbackReason} The same observed relationships remain available below.
        </p>
      )}
      <div className="network-layout" data-component="transaction-intelligence">
        <div className="network-stage" ref={containerRef}>
          {viewMode === 'three' ? (
            <SceneBoundary
              onFailure={() => {
                setDisplay((current) => ({
                  ...current,
                  viewMode: 'fallback',
                  fallbackReason: 'The 3D renderer could not start.',
                }));
              }}
            >
              <Suspense fallback={<Skeleton />}>
                <TransactionNetworkScene
                  network={network}
                  selectedId={selectedId}
                  onSelect={setSelectedId}
                  resetKey={resetKey}
                  active={visible}
                />
              </Suspense>
            </SceneBoundary>
          ) : (
            <NetworkFallback
              transactions={graphTransactions}
              selectedId={selectedId}
              onSelect={setSelectedId}
            />
          )}
        </div>
        <aside className="network-inspector" aria-live="polite">
          {selected ? (
            <>
              <div className="network-inspector-heading">
                <span className="eyebrow">SELECTED TRANSACTION</span>
                <RiskBadge level={selected.prediction.risk.level} />
              </div>
              <h3>{selected.prediction.transaction_id}</h3>
              <DefinitionList
                items={[
                  ['Fused risk score', fmt(selected.prediction.risk.score, 4)],
                  ['Supervised probability', pct(selected.prediction.supervised.probability)],
                  ['Behaviour score', fmt(selected.prediction.behaviour.score, 4)],
                  [
                    'Anomaly diagnostic',
                    `${fmt(selected.prediction.anomaly.score, 4)} · diagnostic only`,
                  ],
                  ['ABRF anomaly contribution', '0'],
                  ['Investigation', selected.investigation.replaceAll('_', ' ')],
                  ['Observed relationships', String(Math.max(0, selectedRelated))],
                  ...relationshipItems,
                ]}
              />
              <div className="network-inspector-actions">
                <Link
                  className="button secondary"
                  to={`/app/transactions/${selected.prediction.transaction_id}`}
                >
                  Transaction detail
                </Link>
                <Link
                  className="text-link"
                  to={`/app/transactions/${selected.prediction.transaction_id}/explanation`}
                >
                  Risk explanation
                </Link>
              </div>
            </>
          ) : (
            <EmptyState
              title="Select a transaction"
              description="Choose a transaction node to inspect its evidence."
            />
          )}
        </aside>
      </div>
      <div className="network-record-picker" aria-label="Select transaction to inspect">
        {graphTransactions.map((transaction) => (
          <button
            key={transaction.prediction.transaction_id}
            type="button"
            className={transaction.prediction.transaction_id === selectedId ? 'selected' : ''}
            aria-pressed={transaction.prediction.transaction_id === selectedId}
            onClick={() => setSelectedId(transaction.prediction.transaction_id)}
          >
            <span className={`risk-dot risk-${transaction.prediction.risk.level.toLowerCase()}`} />
            {transaction.prediction.transaction_id}
          </button>
        ))}
      </div>
      <p className="caption network-caption">
        {network.transactionCount} transaction nodes and {network.edges.length} observed
        relationship edges are shown. Behavioural entity keys are synthetic grouping keys, not
        customer identities.
      </p>
    </Panel>
  );
}
