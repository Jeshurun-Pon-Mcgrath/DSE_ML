import { useState } from 'react';
import { Link, NavLink, useParams } from 'react-router-dom';
import { ArrowUpRight, Save } from 'lucide-react';
import {
  Button,
  DataState,
  DefinitionList,
  PageHeader,
  Panel,
  RiskMeter,
  Toast,
} from '@/components/ui';
import {
  ClassifierEvidence,
  BehaviourEvidence,
  AnomalyEvidence,
  FusionEvidence,
} from '@/features/explanation';
import { states, type ReviewState } from '@/lib/api';
import { useExplanation, useInvestigation, useTransaction } from '@/lib/queries';
import { amount, elapsed, fmt, human, pct } from '@/lib/format';
export default function TransactionPage({ explain = false }: { explain?: boolean }) {
  const { id = '' } = useParams();
  const query = useTransaction(id);
  const explanation = useExplanation(id);
  const mutation = useInvestigation(id);
  const [selected, setSelected] = useState<ReviewState | null>(null);
  return (
    <>
      <PageHeader
        eyebrow="TRANSACTION EVIDENCE"
        title={explain ? 'Risk explanation' : `Transaction ${id}`}
        description={
          explain
            ? 'Four independent explanation layers. Actual attribution and evidence, with their scope made explicit.'
            : 'Review the transaction context, frozen risk decision and supporting intelligence.'
        }
        actions={
          <Link className="button secondary" to="/app/transactions">
            All transactions
          </Link>
        }
      />
      <nav className="tabs" aria-label="Transaction views">
        <NavLink end to={`/app/transactions/${id}`}>
          Transaction detail
        </NavLink>
        <NavLink to={`/app/transactions/${id}/explanation`}>Risk explanation</NavLink>
      </nav>
      {explain ? (
        <DataState query={explanation}>
          {(e) => (
            <div className="page-stack">
              <Panel>
                <RiskMeter
                  score={e.risk_summary.final_risk_score}
                  level={e.risk_summary.risk_level}
                />
              </Panel>
              <ClassifierEvidence explanation={e} />
              <BehaviourEvidence explanation={e} />
              <AnomalyEvidence explanation={e} />
              <FusionEvidence explanation={e} />
              <Panel title="Explanation limitations">
                <ul className="evidence-list">
                  {e.limitations.map((l) => (
                    <li key={l}>{l}</li>
                  ))}
                </ul>
              </Panel>
            </div>
          )}
        </DataState>
      ) : (
        <DataState query={query}>
          {(data) => (
            <div className="page-stack">
              <div className="grid-2">
                <Panel>
                  <RiskMeter
                    score={data.prediction.risk.score}
                    level={data.prediction.risk.level}
                  />
                  <Link className="text-link" to={`/app/transactions/${id}/explanation`}>
                    Inspect full explanation <ArrowUpRight size={14} />
                  </Link>
                </Panel>
                <Panel
                  title="Investigation workflow"
                  note="Analyst decisions do not alter scores, ground truth or final metrics."
                >
                  <DefinitionList
                    items={[
                      ['Current state', human(data.investigation)],
                      [
                        'Model investigation flag',
                        data.prediction.risk.investigation_required
                          ? 'Human review recommended'
                          : 'No flag',
                      ],
                    ]}
                  />
                  <div className="filters">
                    <div className="field">
                      <label htmlFor="investigation-state">Review state</label>
                      <select
                        id="investigation-state"
                        value={selected ?? data.investigation}
                        onChange={(e) => {
                          const next = states.find((s) => s === e.target.value);
                          if (next) setSelected(next);
                        }}
                      >
                        {states.map((s) => (
                          <option key={s} value={s}>
                            {human(s)}
                          </option>
                        ))}
                      </select>
                    </div>
                    <Button
                      onClick={() => mutation.mutate(selected ?? data.investigation)}
                      disabled={mutation.isPending || !selected || selected === data.investigation}
                    >
                      <Save size={14} />
                      Save review
                    </Button>
                  </div>
                  {mutation.isSuccess && (
                    <Toast>Review saved. Model scores remain unchanged.</Toast>
                  )}
                  {mutation.isError && (
                    <p className="field-error" role="alert">
                      {mutation.error.message}
                    </p>
                  )}
                </Panel>
              </div>
              <div className="grid-2">
                <Panel title="Transaction context">
                  <DefinitionList
                    items={Object.entries(data.transaction)
                      .filter(
                        ([key]) =>
                          !['TransactionID', 'TransactionAmt', 'TransactionDT'].includes(key),
                      )
                      .map(([key, v]) => [human(key), v === null ? 'Unavailable' : String(v)])}
                  />
                </Panel>
                <Panel title="Intelligence summary">
                  <DefinitionList
                    items={[
                      [
                        'Amount',
                        amount(
                          typeof data.transaction.TransactionAmt === 'number'
                            ? data.transaction.TransactionAmt
                            : null,
                        ),
                      ],
                      [
                        'Elapsed time',
                        typeof data.transaction.TransactionDT === 'number'
                          ? elapsed(data.transaction.TransactionDT)
                          : 'Unavailable',
                      ],
                      ['Calibrated supervised P', pct(data.prediction.supervised.probability)],
                      ['Behaviour deviation B', fmt(data.prediction.behaviour.score, 4)],
                      ['Behaviour confidence', pct(data.prediction.behaviour.confidence)],
                      ['Prior history count', fmt(data.prediction.behaviour.history_count, 0)],
                      ['Anomaly diagnostic A', fmt(data.prediction.anomaly.score, 4)],
                      ['Anomaly ABRF contribution', '0 · diagnostic only'],
                    ]}
                  />
                  <DataState query={explanation}>
                    {(e) => (
                      <div className="small-gap">
                        <p className="caption">ABRF threshold {e.fusion.decision_threshold}</p>
                        <div className="legend-lines">
                          {e.fusion.reason_codes.map((c) => (
                            <span className="version-tag" key={c}>
                              {human(c)}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </DataState>
                </Panel>
              </div>
            </div>
          )}
        </DataState>
      )}
    </>
  );
}
