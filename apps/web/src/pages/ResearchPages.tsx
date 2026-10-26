import { useState } from 'react';
import { Link } from 'react-router-dom';
import {
  ChartContainer,
  DataState,
  DataTable,
  DefinitionList,
  MetricDisplay,
  Notice,
  PageHeader,
  Panel,
} from '@/components/ui';
import { CorrelationTable, NumericSummary, ResearchData } from '@/features/research';
import {
  FinalComparison,
  FinalMetrics,
  HardNegativePerformance,
  ScenarioPerformance,
} from '@/features/performance';
import { list, number, object, text, valueAt } from '@/lib/api';
import { fmt, human, pct } from '@/lib/format';
import { useModelInfo } from '@/lib/queries';
export function DatasetPage() {
  const [category, setCategory] = useState('merchant_category');
  const model = useModelInfo();
  return (
    <>
      <PageHeader
        eyebrow="SYNTHETIC DATA INTELLIGENCE"
        title="Dataset & exploration"
        description="Understand the development benchmark before interpreting the model. Every observation here is synthetic academic data."
      />
      <div className="page-stack">
        <DataState query={model}>{(m) => <Notice>{m.notice}</Notice>}</DataState>
        <ResearchData path="/dataset/status">
          {(d) => (
            <div className="metric-grid">
              <MetricDisplay label="Development transactions" value={fmt(number(d.row_count), 0)} />
              <MetricDisplay label="Joined columns" value={fmt(number(d.column_count), 0)} />
              <MetricDisplay
                label="Dataset type"
                value={model.data ? human(model.data.dataset_type) : 'Unavailable'}
              />
              <MetricDisplay label="Validation" value={human(text(d.validation_status))} />
            </div>
          )}
        </ResearchData>
        <ResearchData path="/dataset/eda">
          {(d) => {
            const cls = valueAt(d.class_distribution, 'data');
            const missing = object(valueAt(d.missing_data, 'data'));
            const categorical = object(valueAt(d.categorical_frequencies, 'data'));
            const selected = categorical[category];
            return (
              <>
                <div className="grid-2">
                  <ChartContainer
                    title="Development class distribution"
                    note="Configured academic imbalance, not a banking population statistic."
                    data={[
                      { name: 'Legitimate', count: number(valueAt(cls, 'legitimate_count')) },
                      { name: 'Fraud', count: number(valueAt(cls, 'fraud_count')) },
                    ]}
                    x="name"
                    series={[{ key: 'count', label: 'Transactions' }]}
                  />
                  <ChartContainer
                    title="Transaction amount percentiles"
                    note="Quantile summary, not a histogram. Synthetic INR values."
                    data={Object.entries(
                      object(valueAt(d.amount_distribution, 'data', 'percentiles')),
                    ).map(([p, v]) => ({ name: `P${p}`, amount: number(v) }))}
                    x="name"
                    series={[{ key: 'amount', label: 'Amount (INR)' }]}
                  />
                </div>
                <Panel
                  title="Missingness overview"
                  note="Missing identity and context remain unknown. They are not automatically dropped."
                >
                  <DataTable
                    label="Missing values"
                    headers={['Column', 'Missing records', 'Missing %', 'Empty strings']}
                    rows={Object.entries(missing)
                      .sort(
                        (a, b) =>
                          (number(valueAt(b[1], 'percentage')) ?? 0) -
                          (number(valueAt(a[1], 'percentage')) ?? 0),
                      )
                      .filter(([, v]) => (number(valueAt(v, 'count')) ?? 0) > 0)
                      .map(([name, v]) => ({
                        id: name,
                        cells: [
                          human(name),
                          fmt(number(valueAt(v, 'count')), 0),
                          `${fmt(number(valueAt(v, 'percentage')))}%`,
                          fmt(number(valueAt(v, 'empty_string_count')), 0),
                        ],
                      }))}
                  />
                </Panel>
                <div className="field">
                  <label htmlFor="eda-category">Categorical field</label>
                  <select
                    id="eda-category"
                    value={category}
                    onChange={(e) => setCategory(e.target.value)}
                  >
                    {Object.keys(categorical)
                      .filter((k) => !k.endsWith('_id') && k !== 'TransactionID')
                      .map((k) => (
                        <option key={k} value={k}>
                          {human(k)}
                        </option>
                      ))}
                  </select>
                </div>
                <ChartContainer
                  title="Categorical frequencies"
                  note="Bounded most-frequent categories from the development profile."
                  data={list(valueAt(selected, 'frequencies')).map((v) => ({
                    name: text(valueAt(v, 'value')),
                    count: number(valueAt(v, 'count')),
                  }))}
                  x="name"
                  series={[{ key: 'count', label: 'Transactions' }]}
                />
                <NumericSummary
                  data={valueAt(d.numeric_distributions, 'data')}
                  title="Raw numeric distributions"
                />
                <Panel
                  title="Outlier review"
                  note="Outliers are reported, not silently removed. Fraud can itself be unusual."
                >
                  <DataTable
                    label="Outliers"
                    headers={['Feature', 'Finite values', 'IQR outliers', 'Z-score outliers']}
                    rows={Object.entries(object(valueAt(d.outliers, 'data'))).map(([k, v]) => ({
                      id: k,
                      cells: [
                        human(k),
                        fmt(number(valueAt(v, 'finite_count')), 0),
                        fmt(number(valueAt(v, 'iqr_outlier_count')), 0),
                        fmt(number(valueAt(v, 'zscore_outlier_count')), 0),
                      ],
                    }))}
                  />
                </Panel>
              </>
            );
          }}
        </ResearchData>
        <ResearchData path="/statistics/scenarios">
          {(d) => (
            <ChartContainer
              title="Development fraud archetypes"
              note="Evaluation-only ground truth; excluded from model features."
              data={Object.entries(object(d.scenarios)).map(([name, v]) => ({
                name: human(name),
                count: number(valueAt(v, 'count')),
              }))}
              x="name"
              horizontal
              series={[{ key: 'count', label: 'Fraud records' }]}
            />
          )}
        </ResearchData>
        <ResearchData path="/statistics/hard-negatives">
          {(d) => (
            <ChartContainer
              title="Development hard-negative types"
              note="Legitimate unusual records, retained for false-positive evaluation."
              data={Object.entries(object(d.hard_negative_types)).map(([name, v]) => ({
                name: human(name),
                count: number(valueAt(v, 'count')),
              }))}
              x="name"
              horizontal
              series={[{ key: 'count', label: 'Records', color: 'var(--amber)' }]}
            />
          )}
        </ResearchData>
        <ResearchData path="/statistics/correlations">
          {(d) => <CorrelationTable data={d} />}
        </ResearchData>
      </div>
    </>
  );
}
export function StatisticsPage() {
  const [group, setGroup] = useState('all');
  return (
    <>
      <PageHeader
        eyebrow="RESEARCH / DEVELOPMENT"
        title="Statistical analysis"
        description="Descriptive summaries, hypotheses and effect sizes. Synthetic associations explain the benchmark, not real-world causation."
      />
      <div className="page-stack">
        <Notice>
          Training-partition hypothesis tests use nominal exploratory inference. Repeated entities
          and attack episodes violate independent-row assumptions; statistical significance is not
          operational importance.
        </Notice>
        <ResearchData path="/statistics/hypotheses">
          {(d) => (
            <>
              <div className="metric-grid">
                <MetricDisplay label="Research questions" value={fmt(number(d.family_size), 0)} />
                <MetricDisplay
                  label="Comparison correction"
                  value="BH FDR"
                  note={text(d.correction)}
                />
              </div>
              <div className="grid-2">
                {list(d.hypotheses).map((v) => {
                  const h = object(v),
                    out = object(h.outcome);
                  return (
                    <Panel
                      key={text(h.id)}
                      title={`${text(h.id)} · ${human(text(h.feature))}`}
                      note={text(h.question)}
                    >
                      <p className="caption">
                        <strong>H₀</strong> {text(h.h0)}
                      </p>
                      <p className="caption">
                        <strong>H₁</strong> {text(h.h1)}
                      </p>
                      <DefinitionList
                        items={[
                          ['Test', text(out.method)],
                          ['Raw p-value', number(out.p_value)?.toExponential(3) ?? 'Unavailable'],
                          [
                            'Adjusted p-value',
                            number(h.adjusted_p_value)?.toExponential(3) ?? 'Unavailable',
                          ],
                          ['Effect size', fmt(number(valueAt(out.effect_size, 'value')), 4)],
                          ['Effect method', text(valueAt(out.effect_size, 'method'))],
                          ['Decision', human(text(h.decision))],
                        ]}
                      />
                      <p className="caption">{text(h.practical_interpretation)}</p>
                      <details className="research-card">
                        <summary>Assumptions and reasoning</summary>
                        <p className="caption">{text(h.test_selection_reasoning)}</p>
                        <ul className="evidence-list">
                          {list(out.assumption_checks).map((s, i) => (
                            <li key={i}>{text(s)}</li>
                          ))}
                        </ul>
                      </details>
                    </Panel>
                  );
                })}
              </div>
            </>
          )}
        </ResearchData>
        <ResearchData path="/statistics/summary">
          {(d) => {
            const groups = object(valueAt(d.descriptive, 'groups'));
            return (
              <>
                <div className="field">
                  <label htmlFor="stat-group">Descriptive population</label>
                  <select id="stat-group" value={group} onChange={(e) => setGroup(e.target.value)}>
                    {Object.keys(groups).map((g) => (
                      <option key={g} value={g}>
                        {human(g)}
                      </option>
                    ))}
                  </select>
                </div>
                <NumericSummary data={groups[group]} title="Descriptive statistics" />
              </>
            );
          }}
        </ResearchData>
        <ResearchData path="/statistics/correlations">
          {(d) => <CorrelationTable data={d} />}
        </ResearchData>
      </div>
    </>
  );
}
export function PerformancePage({
  view = 'models',
}: {
  view?: 'models' | 'scenarios' | 'hard-negatives';
}) {
  return (
    <>
      <PageHeader
        eyebrow="FROZEN EVALUATION"
        title={
          view === 'models'
            ? 'Model performance'
            : view === 'scenarios'
              ? 'Fraud scenario performance'
              : 'Hard-negative performance'
        }
        description="Final holdout results are authoritative. Earlier validation metrics remain clearly marked as development research."
      />
      <nav className="tabs" aria-label="Evaluation views">
        {[
          ['models', '/app/performance', 'Models & final holdout'],
          ['scenarios', '/app/performance/scenarios', 'Scenario recall'],
          ['hard-negatives', '/app/performance/hard-negatives', 'Hard negatives'],
        ].map(([key, path, label]) => (
          <Link key={key} className={view === key ? 'active' : ''} to={path ?? '/app/performance'}>
            {label}
          </Link>
        ))}
      </nav>
      <div className="page-stack">
        {view === 'scenarios' ? (
          <ScenarioPerformance />
        ) : view === 'hard-negatives' ? (
          <>
            <FinalMetrics />
            <HardNegativePerformance />
          </>
        ) : (
          <>
            <FinalComparison />
            <ResearchData path="/models/comparison">
              {(d) => (
                <>
                  <Panel
                    title="Classifier comparison · Validation / Development"
                    note="This table was used during development. It is not final holdout performance."
                  >
                    <DataTable
                      label="Development classifiers"
                      headers={[
                        'Algorithm',
                        'Status',
                        'Precision',
                        'Recall',
                        'F1',
                        'PR-AUC',
                        'ROC-AUC',
                      ]}
                      rows={list(d.rows).map((v) => {
                        const r = object(v),
                          m = object(r.validation_metrics);
                        return {
                          id: text(r.model_id),
                          cells: [
                            `${text(r.algorithm)}${r.is_selected_supervised_model === true ? ' · SELECTED' : ''}`,
                            text(r.status),
                            pct(number(m.precision)),
                            pct(number(m.recall)),
                            fmt(number(m.f1), 4),
                            fmt(number(m.pr_auc), 4),
                            fmt(number(m.roc_auc), 4),
                          ],
                        };
                      })}
                    />
                  </Panel>
                  <Panel title="Selection methodology">
                    <p className="caption">{text(d.selection_rationale)}</p>
                    <p className="caption">{text(d.note)}</p>
                    {list(d.rows)
                      .map((v) => object(v))
                      .filter((r) => r.limitation !== null)
                      .map((r) => (
                        <p key={text(r.model_id)} className="caption">
                          {text(r.algorithm)}: {text(r.limitation)}
                        </p>
                      ))}
                  </Panel>
                </>
              )}
            </ResearchData>
          </>
        )}
      </div>
    </>
  );
}
