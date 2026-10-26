import { DataTable, DefinitionList, EmptyState, Notice, Panel } from '@/components/ui';
import type { Explanation, Feature } from '@/lib/api';
import { fmt, human, pct } from '@/lib/format';
export function ShapFactors({
  features,
  direction,
  scale,
}: {
  features: Feature[];
  direction: 'positive' | 'negative';
  scale?: number;
}) {
  const max = scale ?? Math.max(...features.map((f) => Math.abs(f.shap_value)), 0.000001);
  return features.length ? (
    <div className="feature-list">
      {features.map((f) => (
        <article className={`feature-row ${direction}`} key={f.feature}>
          <div className="feature-title">
            <strong>{f.display_name}</strong>
            <span className="tabular">
              {f.shap_value > 0 ? '+' : ''}
              {fmt(f.shap_value, 5)}
            </span>
          </div>
          <div className="feature-importance">
            <span style={{ width: `${(Math.abs(f.shap_value) / max) * 100}%` }} />
          </div>
          <p>{f.statement}</p>
          <p className="caption">
            Model input {fmt(f.feature_value, 4)} · {human(f.interpretability)} ·{' '}
            {human(f.feature_family)}
            {f.raw_or_derived ? ` · ${human(f.raw_or_derived)}` : ''}
          </p>
        </article>
      ))}
    </div>
  ) : (
    <EmptyState
      title="No factors in this direction"
      description="This is not a zero-attribution claim; consult the SHAP availability status."
    />
  );
}
export function ClassifierEvidence({ explanation: e }: { explanation: Explanation }) {
  const scale = Math.max(
    ...[...e.supervised.top_positive_features, ...e.supervised.top_negative_features].map((f) =>
      Math.abs(f.shap_value),
    ),
    0.000001,
  );
  return (
    <div className="page-stack">
      <Notice>
        Tree SHAP explains the underlying Decision Tree. Sigmoid calibration and the full ABRF
        system are outside this decomposition. Attribution does not establish causation or intent.
        The strongest factors below are a truncated view of the complete attribution.
      </Notice>
      <div className="grid-2">
        <Panel
          title="Risk-increasing classifier factors"
          note="Positive SHAP contributions to the underlying fraud-class output."
        >
          <ShapFactors
            features={e.supervised.top_positive_features}
            direction="positive"
            scale={scale}
          />
        </Panel>
        <Panel
          title="Risk-decreasing classifier factors"
          note="Negative SHAP contributions; no guarantee of legitimacy."
        >
          <ShapFactors
            features={e.supervised.top_negative_features}
            direction="negative"
            scale={scale}
          />
        </Panel>
      </div>
      <Panel title="Classifier output scope">
        <DefinitionList
          items={[
            ['Explained model', human(e.supervised.explained_model)],
            ['Raw tree probability', pct(e.supervised.raw_probability)],
            ['Calibrated supervised probability P', pct(e.supervised.calibrated_probability)],
            ['SHAP expected value', fmt(e.supervised.base_value, 6)],
            ['SHAP status', e.supervised.shap_status],
            ['Model version', e.supervised.model_version],
          ]}
        />
      </Panel>
    </div>
  );
}
export function BehaviourEvidence({ explanation: e }: { explanation: Explanation }) {
  return (
    <div className="page-stack">
      <div className="grid-2">
        <Panel title="Strictly prior behavioural evidence">
          <DefinitionList
            items={[
              ['Deviation score B', fmt(e.behaviour.behaviour_score, 4)],
              ['Behaviour confidence', pct(e.behaviour.behaviour_confidence)],
              ['Prior history count', fmt(e.behaviour.history_count, 0)],
              ['History available', e.behaviour.history_available ? 'Yes' : 'No, cold start'],
            ]}
          />
          {!e.behaviour.history_available && (
            <Notice>
              Missing history is uncertainty, not evidence of low risk. Behaviour may be
              unavailable.
            </Notice>
          )}
        </Panel>
        <Panel
          title="Confidence measurements"
          note="Confidence describes history reliability, not likelihood of fraud."
        >
          <ul className="evidence-list">
            {e.behaviour.confidence_explanation.supporting_measurements.map((m) => (
              <li key={m}>{m}</li>
            ))}
          </ul>
        </Panel>
      </div>
      <Panel
        title="Behavioural components"
        note="Deterministic evidence, ranked by contribution. Current and future events are excluded from prior history."
      >
        <DataTable
          label="Behavioural evidence"
          headers={['Signal', 'Observed value', 'Component score', 'Contribution', 'Availability']}
          rows={[...e.behaviour.components]
            .sort((a, b) => (b.contribution ?? -1) - (a.contribution ?? -1))
            .map((c) => ({
              id: c.signal,
              cells: [
                human(c.signal),
                fmt(c.raw_value, 4),
                fmt(c.score, 4),
                fmt(c.contribution, 4),
                c.available ? 'Available' : 'Unavailable',
              ],
            }))}
        />
      </Panel>
      <Panel title="Entity event history">
        <EmptyState
          title="Event-level history is not exposed"
          description="The current API provides the stored strictly-prior feature summary, not a complete entity timeline. No historical events are reconstructed or invented."
        />
      </Panel>
    </div>
  );
}
export function AnomalyEvidence({ explanation: e }: { explanation: Explanation }) {
  return (
    <Panel
      title="Anomaly diagnostic"
      note="Isolation Forest identifies unusual observations; it does not estimate fraud probability."
    >
      <Notice>
        Diagnostic only. ABRF contribution = 0. Validation selected zero anomaly weight in the
        active configuration.
      </Notice>
      <DefinitionList
        items={[
          ['Anomaly score A', fmt(e.anomaly.score, 4)],
          ['Frozen anomaly threshold', fmt(e.anomaly.threshold, 6)],
          ['Diagnostic flag', e.anomaly.flag ? 'Flagged as unusual' : 'Not flagged'],
          ['Effective ABRF weight', fmt(e.anomaly.effective_abrf_weight, 1)],
          ['Influenced final risk', e.anomaly.influenced_final_risk ? 'Yes' : 'No'],
        ]}
      />
    </Panel>
  );
}
export function FusionEvidence({ explanation: e }: { explanation: Explanation }) {
  return (
    <div className="page-stack">
      <Panel
        title="Exact ABRF calculation"
        note="RiskByte fused risk score. Not a literal fraud probability."
      >
        <div className="formula">
          R = (wₚP + wₐA + wᵦg(Cᵦ)B) / (wₚ + wₐ + wᵦg(Cᵦ))
          <small>{e.formula_trace.confidence_rule}</small>
        </div>
        <p className="caption mt-3">
          Displayed values are rounded. The API preserves the full-precision trace.
        </p>
        <DefinitionList
          items={[
            ['P · calibrated supervised probability', fmt(e.fusion.supervised_probability, 6)],
            ['A · diagnostic anomaly score', fmt(e.fusion.anomaly_score, 6)],
            ['B · behavioural deviation', fmt(e.fusion.behaviour_score, 6)],
            ['Cᵦ · behavioural confidence', fmt(e.fusion.behaviour_confidence, 6)],
            ['g(Cᵦ)', fmt(e.formula_trace.transformed_confidence, 6)],
            ['Renormalization denominator', fmt(e.formula_trace.denominator, 6)],
            ['Final fused risk', fmt(e.fusion.final_risk_score, 8)],
            ['Investigation threshold', String(e.fusion.decision_threshold)],
            [
              'Investigation required',
              e.fusion.investigation_flag
                ? 'Yes, human review recommended'
                : 'No flag under the frozen policy',
            ],
          ]}
        />
      </Panel>
      <Panel title="Evidence contribution trace">
        <DataTable
          label="ABRF weights and contributions"
          headers={['Component', 'Base weight', 'Effective weight', 'Weighted contribution']}
          rows={(['supervised', 'anomaly', 'behaviour'] as const).map((key, i) => ({
            id: key,
            cells: [
              human(key),
              fmt(e.fusion.base_weights[['wp', 'wa', 'wb'][i] ?? 'wp'], 3),
              fmt(e.fusion.effective_weights[key], 6),
              fmt(e.fusion.weighted_contributions[key], 8),
            ],
          }))}
        />
      </Panel>
      <Panel title="Deterministic reason codes">
        {e.fusion.reason_codes.length ? (
          <div className="legend-lines">
            {e.fusion.reason_codes.map((code) => (
              <span className="version-tag" key={code}>
                {human(code)}
              </span>
            ))}
          </div>
        ) : (
          <p className="caption">No configured reason code crossed its evidence threshold.</p>
        )}
      </Panel>
    </div>
  );
}
