import { Link } from 'react-router-dom';
import {
  DataState,
  DataTable,
  MetricDisplay,
  Panel,
  ChartContainer,
  Notice,
} from '@/components/ui';
import { useFinalEvaluation, useScenarios, useHardNegatives } from '@/lib/queries';
import { fmt, human, pct } from '@/lib/format';
export function FinalMetrics() {
  const query = useFinalEvaluation();
  return (
    <DataState query={query}>
      {(data) => (
        <div className="small-gap">
          <div className="metric-grid">
            <MetricDisplay
              label="Final ABRF precision"
              value={pct(data.abrf.precision)}
              note="Frozen future holdout"
            />
            <MetricDisplay
              label="Final ABRF recall"
              value={pct(data.abrf.recall)}
              note="Fraud cases detected"
            />
            <MetricDisplay
              label="Final ABRF F1"
              value={fmt(data.abrf.f1, 4)}
              note="Precision / recall balance"
            />
            <MetricDisplay
              label="Final ABRF PR-AUC"
              value={fmt(data.abrf.pr_auc, 4)}
              note="Average precision"
            />
          </div>
          <p className="caption">
            One locked evaluation · {fmt(data.transaction_count, 0)} synthetic transactions ·{' '}
            {fmt(data.fraud_count, 0)} fraud cases.{' '}
            <Link className="text-link" to="/app/performance">
              View full results
            </Link>
          </p>
        </div>
      )}
    </DataState>
  );
}
export function FinalComparison() {
  const query = useFinalEvaluation();
  return (
    <DataState query={query}>
      {(data) => (
        <div className="page-stack">
          <Panel
            title="Final future holdout"
            note="Authoritative results. Models and thresholds were frozen before this one-use evaluation."
          >
            <DataTable
              label="Final model comparison"
              headers={[
                'System',
                'Precision',
                'Recall',
                'F1',
                'PR-AUC',
                'ROC-AUC',
                'FPR',
                'Hard-negative FPR',
              ]}
              rows={(
                [
                  ['Frozen supervised', data.supervised],
                  ['Final ABRF', data.abrf],
                ] as const
              ).map(([name, v]) => {
                return {
                  id: String(name),
                  cells: [
                    String(name),
                    pct(v.precision),
                    pct(v.recall),
                    fmt(v.f1, 4),
                    fmt(v.pr_auc, 4),
                    fmt(v.roc_auc, 4),
                    pct(v.fpr, 4),
                    pct(v.hard_negatives.rate),
                  ],
                };
              })}
            />
          </Panel>
          <div className="grid-2">
            <Panel
              title="Confusion matrix"
              note="Final ABRF decisions against synthetic evaluation labels."
            >
              <DataTable
                label="ABRF confusion matrix"
                headers={['', 'Predicted legitimate', 'Investigation flag']}
                rows={[
                  {
                    id: 'legit',
                    cells: [
                      'Actually legitimate',
                      fmt(data.abrf.confusion.tn, 0),
                      fmt(data.abrf.confusion.fp, 0),
                    ],
                  },
                  {
                    id: 'fraud',
                    cells: [
                      'Actually fraud',
                      fmt(data.abrf.confusion.fn, 0),
                      fmt(data.abrf.confusion.tp, 0),
                    ],
                  },
                ]}
              />
            </Panel>
            <Panel title="Detection tradeoff">
              <div className="metric-grid">
                <MetricDisplay
                  label="Additional fraud recovered"
                  value={fmt(data.abrf.confusion.tp - data.supervised.confusion.tp, 0)}
                />
                <MetricDisplay
                  label="Additional false positives"
                  value={fmt(data.abrf.confusion.fp - data.supervised.confusion.fp, 0)}
                />
              </div>
              <p className="caption">
                Higher recall comes with lower precision. Different frozen operating thresholds also
                contribute to the change; behavioural evidence alone did not cause every additional
                detection.
              </p>
            </Panel>
          </div>
          <Notice>
            All results describe synthetic academic transactions. They cannot establish production
            banking performance.
          </Notice>
        </div>
      )}
    </DataState>
  );
}
export function ScenarioPerformance() {
  const query = useScenarios();
  return (
    <DataState query={query}>
      {(data) => (
        <div className="page-stack">
          <ChartContainer
            title="Final recall by fraud scenario"
            note="Weak results remain visible. Scenario metadata is evaluation-only."
            data={Object.entries(data).map(([name, v]) => ({
              name: human(name),
              supervised: v.supervised.rate,
              abrf: v.abrf.rate,
            }))}
            x="name"
            horizontal
            percent
            series={[
              { key: 'supervised', label: 'Frozen supervised', color: 'var(--slate)' },
              { key: 'abrf', label: 'Final ABRF' },
            ]}
          />
          <Panel title="Scenario evidence">
            <DataTable
              label="Final scenario performance"
              headers={[
                'Scenario',
                'Count',
                'Supervised recall',
                'ABRF recall',
                'Diagnostic anomaly recall',
              ]}
              rows={Object.entries(data).map(([name, v]) => ({
                id: name,
                cells: [
                  human(name),
                  fmt(v.count, 0),
                  pct(v.supervised.rate),
                  pct(v.abrf.rate),
                  pct(v.anomaly.rate),
                ],
              }))}
            />
          </Panel>
        </div>
      )}
    </DataState>
  );
}
export function HardNegativePerformance() {
  const query = useHardNegatives();
  return (
    <DataState query={query}>
      {(data) => (
        <div className="page-stack">
          <ChartContainer
            title="Legitimate unusual transactions"
            note="Hard negatives deliberately resemble suspicious activity. A flag is a false positive here."
            data={Object.entries(data).map(([name, v]) => ({
              name: human(name),
              supervised: v.supervised.rate,
              abrf: v.abrf.rate,
            }))}
            x="name"
            horizontal
            percent
            series={[
              { key: 'supervised', label: 'Supervised FPR', color: 'var(--slate)' },
              { key: 'abrf', label: 'ABRF FPR', color: 'var(--amber)' },
            ]}
          />
          <Panel title="Hard-negative protection">
            <DataTable
              label="Hard-negative results"
              headers={[
                'Type',
                'Transactions',
                'Supervised false positives',
                'ABRF false positives',
                'ABRF FPR',
              ]}
              rows={Object.entries(data).map(([name, v]) => ({
                id: name,
                cells: [
                  human(name),
                  fmt(v.count, 0),
                  fmt(v.supervised.flagged, 0),
                  fmt(v.abrf.flagged, 0),
                  pct(v.abrf.rate),
                ],
              }))}
            />
          </Panel>
        </div>
      )}
    </DataState>
  );
}
