import { Link, useSearchParams } from 'react-router-dom';
import {
  AnomalyEvidence,
  BehaviourEvidence,
  ClassifierEvidence,
  FusionEvidence,
} from '@/features/explanation';
import { GroupDistribution, ResearchData } from '@/features/research';
import {
  ChartContainer,
  DataState,
  DataTable,
  EmptyState,
  DefinitionList,
  PageHeader,
  Panel,
  Notice,
} from '@/components/ui';
import { list, number, object, text, valueAt } from '@/lib/api';
import { useExplanation, useTransactions } from '@/lib/queries';
import { fmt, human, pct } from '@/lib/format';
export default function IntelligencePage({
  kind,
}: {
  kind: 'behaviour' | 'anomaly' | 'abrf' | 'explainability';
}) {
  const [params, setParams] = useSearchParams();
  const transactions = useTransactions({ page: '1', page_size: '100', sort: 'recent' });
  const id =
    params.get('transaction') ?? transactions.data?.items[0]?.prediction.transaction_id ?? '';
  const explanation = useExplanation(id);
  const title = {
    behaviour: 'Behaviour intelligence',
    anomaly: 'Anomaly intelligence',
    abrf: 'ABRF fusion',
    explainability: 'Explainable intelligence',
  }[kind];
  return (
    <>
      <PageHeader
        eyebrow="INTELLIGENCE / EVIDENCE"
        title={title}
        description={
          {
            behaviour:
              'Deviation from an entity’s own strictly prior activity, with confidence and cold-start handling.',
            anomaly:
              'Isolation Forest diagnostics remain separate from fraud probability and active fusion.',
            abrf: 'A transparent, confidence-aware calculation using frozen supervised and behavioural evidence.',
            explainability:
              'Inspect genuine classifier attributions and distinguish them from behavioural and fusion evidence.',
          }[kind]
        }
      />
      <div className="page-stack">
        {kind === 'anomaly' && (
          <Notice>
            Anomaly detection is retained as diagnostic evidence. Validation selected zero anomaly
            weight in the active ABRF configuration.
          </Notice>
        )}
        {kind === 'abrf' && (
          <ResearchData path="/abrf/config">
            {(d) => {
              const c = object(d.configuration);
              return (
                <Panel
                  title="Frozen active configuration"
                  note="Selected using validation only. No controls on this page modify the fusion."
                >
                  <div className="formula">
                    R = (wₚP + wₐA + wᵦg(Cᵦ)B) / (wₚ + wₐ + wᵦg(Cᵦ))
                    <small>
                      g(Cᵦ) = Cᵦ when Cᵦ ≥ 0.5; otherwise zero. Unavailable B is excluded.
                    </small>
                  </div>
                  <DefinitionList
                    items={[
                      ['ABRF version', text(c.version)],
                      ['Base supervised weight P', fmt(number(valueAt(c.weights, 'wp')), 1)],
                      ['Base anomaly weight A', fmt(number(valueAt(c.weights, 'wa')), 1)],
                      ['Base behaviour weight B', fmt(number(valueAt(c.weights, 'wb')), 1)],
                      [
                        'Investigation threshold',
                        number(c.decision_threshold)?.toString() ?? 'Unavailable',
                      ],
                      ['Operating policy', text(c.policy)],
                    ]}
                  />
                  <div className="legend-lines">
                    {list(c.risk_boundaries).map((v, i) => (
                      <span key={i} className="version-tag">
                        Boundary {i + 1}: {fmt(number(v), 6)}
                      </span>
                    ))}
                  </div>
                  <p className="caption">
                    Risk bands are decision-support categories, not statements of guilt. Anomaly
                    remains architecturally available but has zero active contribution.
                  </p>
                </Panel>
              );
            }}
          </ResearchData>
        )}
        {kind === 'explainability' && (
          <ResearchData path="/explainability/global">
            {(d) => (
              <>
                <ChartContainer
                  title="Global classifier SHAP importance"
                  note="Validation / Development. Mean absolute SHAP, underlying tree only."
                  data={list(d.importance)
                    .slice(0, 10)
                    .map((v) => ({
                      name: human(text(valueAt(v, 'feature'))),
                      importance: number(valueAt(v, 'mean_absolute_shap')),
                    }))}
                  x="name"
                  horizontal
                  series={[{ key: 'importance', label: 'Mean absolute SHAP' }]}
                />
                <Panel title="Feature interpretation">
                  <DataTable
                    label="SHAP feature metadata"
                    headers={['Feature', 'Family', 'Interpretability', 'Mean absolute SHAP']}
                    rows={list(d.importance)
                      .slice(0, 10)
                      .map((v) => ({
                        id: text(valueAt(v, 'feature')),
                        cells: [
                          human(text(valueAt(v, 'feature'))),
                          human(text(valueAt(v, 'feature_family'))),
                          human(text(valueAt(v, 'interpretability'))),
                          fmt(number(valueAt(v, 'mean_absolute_shap')), 6),
                        ],
                      }))}
                  />
                </Panel>
              </>
            )}
          </ResearchData>
        )}
        {(kind === 'behaviour' || kind === 'anomaly') && (
          <ResearchData path="/behaviour/summary">
            {(d) => (
              <GroupDistribution
                data={d.groups}
                signal={kind === 'behaviour' ? 'behaviour_score' : 'anomaly_score'}
                title={`${human(kind)} score distributions · Validation / Development`}
              />
            )}
          </ResearchData>
        )}
        {kind === 'behaviour' && (
          <ResearchData path="/behaviour/summary">
            {(d) => (
              <GroupDistribution
                data={d.groups}
                signal="behaviour_confidence"
                title="History confidence · Validation / Development"
              />
            )}
          </ResearchData>
        )}
        {kind === 'anomaly' && (
          <ResearchData path="/anomaly/metrics">
            {(d) => (
              <Panel
                title="Validation diagnostic performance"
                note="Development metrics, not headline final results. A flags unusual activity, not fraud guilt."
              >
                <DefinitionList
                  items={[
                    ['PR-AUC', fmt(number(d.pr_auc), 4)],
                    ['ROC-AUC', fmt(number(d.roc_auc), 4)],
                    ['Fraud recall', pct(number(d.recall))],
                    ['Legitimate FPR', pct(number(d.fpr))],
                    ['Hard-negative FPR', pct(number(valueAt(d.hard_negatives, 'rate')))],
                  ]}
                />
              </Panel>
            )}
          </ResearchData>
        )}
        {kind === 'anomaly' && (
          <ResearchData path="/anomaly/scenarios">
            {(d) => (
              <Panel
                title="Scenario diagnostics · Validation / Development"
                note="These are historical validation diagnostics, not final holdout results."
              >
                <DataTable
                  label="Validation anomaly scenarios"
                  headers={['Scenario', 'Fraud observations', 'Diagnostic recall']}
                  rows={Object.entries(object(d.anomaly)).map(([name, v]) => ({
                    id: name,
                    cells: [
                      human(name),
                      fmt(number(valueAt(v, 'count')), 0),
                      pct(number(valueAt(v, 'rate'))),
                    ],
                  }))}
                />
                <Link className="text-link mt-4" to="/app/performance/scenarios">
                  View final scenario performance
                </Link>
              </Panel>
            )}
          </ResearchData>
        )}
        <Panel
          title="Transaction evidence"
          note="Select an actual observed record. No history or signals are invented."
        >
          <DataState query={transactions}>
            {(data) =>
              data.items.length === 0 ? (
                <EmptyState
                  title="No observed transactions"
                  description="Submit a synthetic transaction through the transaction explorer to inspect its evidence."
                />
              ) : (
                <div className="field">
                  <label htmlFor="evidence-transaction">Transaction</label>
                  <select
                    id="evidence-transaction"
                    value={id}
                    onChange={(e) => setParams({ transaction: e.target.value })}
                  >
                    {id && !data.items.some((t) => t.prediction.transaction_id === id) && (
                      <option value={id}>{id} · Requested transaction</option>
                    )}
                    {data.items.map((t) => (
                      <option key={t.prediction.transaction_id} value={t.prediction.transaction_id}>
                        {t.prediction.transaction_id} · {t.prediction.risk.level}
                      </option>
                    ))}
                  </select>
                  <p className="caption">
                    Showing {id}. The picker lists the latest 100 records; use the transaction
                    explorer for other records.
                  </p>
                  <Link className="text-link" to={`/app/transactions/${id}/explanation`}>
                    Open all explanation layers
                  </Link>
                </div>
              )
            }
          </DataState>
        </Panel>
        {id && (
          <DataState query={explanation}>
            {(e) =>
              kind === 'behaviour' ? (
                <BehaviourEvidence explanation={e} />
              ) : kind === 'anomaly' ? (
                <AnomalyEvidence explanation={e} />
              ) : kind === 'abrf' ? (
                <FusionEvidence explanation={e} />
              ) : (
                <ClassifierEvidence explanation={e} />
              )
            }
          </DataState>
        )}
      </div>
    </>
  );
}
