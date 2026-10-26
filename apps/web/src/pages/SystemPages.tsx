import { useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import {
  Button,
  DataState,
  DataTable,
  DefinitionList,
  Notice,
  PageHeader,
  Pagination,
  Panel,
  StatusIndicator,
  Toast,
} from '@/components/ui';
import { useAudit, useHealth, useModelInfo } from '@/lib/queries';
import { human } from '@/lib/format';
import { env } from '@/config/env';
import { SYNTHETIC_DATASET_NOTICE } from '@/config/dataset';
export function SystemPage() {
  const model = useModelInfo(),
    health = useHealth();
  return (
    <>
      <PageHeader
        eyebrow="PROVENANCE / READINESS"
        title="System & model"
        description="Frozen component versions, evaluation status and actual service readiness."
      />
      <div className="page-stack">
        <DataState query={model}>
          {(m) => (
            <>
              <Notice>{m.notice}</Notice>
              <div className="grid-2">
                <Panel title="Frozen intelligence stack">
                  <DefinitionList
                    items={[
                      ['Supervised model', m.supervised_model],
                      ['Model identifier', m.supervised_model_id],
                      ['Model version', m.model_version],
                      ['Calibration', m.supervised_calibration],
                      ['Feature pipeline', m.feature_version],
                      ['Anomaly version', m.anomaly_version],
                      ['Behaviour version', m.behaviour_version],
                      ['Fusion version', m.abrf_version],
                      ['Explainability', m.explainability_version],
                      ['SHAP library', m.shap_version],
                    ]}
                  />
                </Panel>
                <Panel title="Dataset integrity">
                  <DefinitionList
                    items={[
                      ['Dataset type', m.dataset_type],
                      ['Final evaluation state', m.final_holdout_status],
                      ['System status', m.system_status],
                      ['Minimum accepted elapsed time', `${m.minimum_elapsed_seconds} seconds`],
                    ]}
                  />
                  <p className="eyebrow mt-6">DEVELOPMENT FINGERPRINT</p>
                  <p className="mono-block">{m.development_fingerprint}</p>
                  <p className="eyebrow mt-6">FINAL HOLDOUT FINGERPRINT</p>
                  <p className="mono-block">{m.final_holdout_fingerprint}</p>
                </Panel>
              </div>
            </>
          )}
        </DataState>
        <DataState query={health}>
          {(h) => (
            <Panel title="Service readiness">
              <DefinitionList
                items={Object.entries(h)
                  .filter(([k]) => k !== 'unavailable_reason')
                  .map(([key, value]) => [
                    human(key),
                    <StatusIndicator
                      healthy={value === true}
                      label={value === true ? 'Ready' : 'Unavailable'}
                    />,
                  ])}
              />
              {h.unavailable_reason && <p className="caption">{h.unavailable_reason}</p>}
            </Panel>
          )}
        </DataState>
      </div>
    </>
  );
}
export function AuditPage() {
  const [page, setPage] = useState(1),
    [id, setId] = useState('');
  const query = useAudit(page, id);
  return (
    <>
      <PageHeader
        eyebrow="OPERATIONS / TRACEABILITY"
        title="Audit history"
        description="Durable prediction and investigation events. Financial context and credentials are excluded."
      />
      <div className="filters">
        <div className="field">
          <label htmlFor="audit-id">Exact transaction ID</label>
          <input
            id="audit-id"
            value={id}
            maxLength={100}
            onChange={(e) => {
              setId(e.target.value);
              setPage(1);
            }}
            type="search"
          />
        </div>
      </div>
      <DataState query={query}>
        {(d) => (
          <Panel>
            <DataTable
              label="Audit events"
              headers={['Event', 'Transaction', 'Timestamp (UTC)', 'Safe review metadata']}
              rows={d.items.map((e) => ({
                id: String(e.id),
                cells: [
                  human(e.action),
                  <Link to={`/app/transactions/${e.transaction_id}`}>{e.transaction_id}</Link>,
                  new Date(
                    e.timestamp.endsWith('Z') || /[+-]\d\d:\d\d$/.test(e.timestamp)
                      ? e.timestamp
                      : `${e.timestamp}Z`,
                  ).toLocaleString('en-GB', { timeZone: 'UTC', hour12: false }),
                  Object.entries(e.metadata)
                    .map(
                      ([k, v]) => `${human(k)}: ${typeof v === 'string' ? v : JSON.stringify(v)}`,
                    )
                    .join(' · ') || 'No additional metadata',
                ],
              }))}
            />
            <Pagination page={page} total={d.total} pageSize={25} onChange={setPage} />
          </Panel>
        )}
      </DataState>
    </>
  );
}
export function SettingsPage() {
  const [compact, setCompact] = useState(localStorage.getItem('riskbyte.compact') === 'true');
  const [pageSize, setPageSize] = useState(localStorage.getItem('riskbyte.pageSize') ?? '15');
  const [saved, setSaved] = useState(false);
  const client = useQueryClient();
  const health = useHealth();
  return (
    <>
      <PageHeader
        eyebrow="WORKSPACE PREFERENCES"
        title="Settings"
        description="Local display preferences and connection details. Frozen model configuration is read-only."
      />
      <div className="page-stack">
        <Panel
          title="Display preferences"
          note="Saved only in this browser. No model, threshold or score is affected."
        >
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={compact}
              onChange={(e) => {
                setCompact(e.target.checked);
                setSaved(false);
              }}
            />
            Compact table density
          </label>
          <div className="field">
            <label htmlFor="page-size">Transactions per page</label>
            <select
              id="page-size"
              value={pageSize}
              onChange={(e) => {
                setPageSize(e.target.value);
                setSaved(false);
              }}
            >
              <option>15</option>
              <option>25</option>
              <option>50</option>
            </select>
          </div>
          <div className="form-actions">
            <Button
              onClick={() => {
                localStorage.setItem('riskbyte.compact', String(compact));
                localStorage.setItem('riskbyte.pageSize', pageSize);
                document.documentElement.dataset.density = compact ? 'compact' : 'comfortable';
                setSaved(true);
              }}
            >
              Save preferences
            </Button>
          </div>
          {saved && <Toast>Workspace preferences saved in this browser.</Toast>}
        </Panel>
        <Panel title="API connection">
          <DefinitionList
            items={[
              ['Configured API origin', env.VITE_API_BASE_URL],
              ['Development transport', 'Same-origin Vite proxy'],
              ['Write authentication', 'Trusted operator proxy in production'],
              ['Model controls', 'Read-only, frozen'],
            ]}
          />
          <Button
            className="secondary"
            onClick={() => {
              void client.invalidateQueries();
            }}
          >
            Refresh API data
          </Button>
          <DataState query={health}>
            {(h) => (
              <StatusIndicator
                healthy={h.inference_ready}
                label={h.inference_ready ? 'Inference ready' : 'Inference unavailable'}
              />
            )}
          </DataState>
        </Panel>
        <Panel title="About this environment">
          <p className="caption">{SYNTHETIC_DATASET_NOTICE}</p>
          <p className="caption">
            No enterprise sign-in or user-management interface is provided. Risk flags recommend
            review and do not determine fraud guilt.
          </p>
          <div className="legend-lines">
            <Link className="text-link" to="/privacy">
              Privacy policy
            </Link>
            <Link className="text-link" to="/terms">
              Terms and conditions
            </Link>
            <Link className="text-link" to="/app/system">
              Model provenance
            </Link>
          </div>
        </Panel>
      </div>
    </>
  );
}
