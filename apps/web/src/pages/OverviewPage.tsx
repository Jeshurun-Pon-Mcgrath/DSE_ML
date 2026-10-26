import { Activity, ArrowUpRight, ListFilter, ShieldCheck, SlidersHorizontal } from 'lucide-react';
import { Link } from 'react-router-dom';
import {
  ChartContainer,
  DataState,
  MetricDisplay,
  PageHeader,
  Panel,
  SectionHeader,
} from '@/components/ui';
import { FinalMetrics } from '@/features/performance';
import { TransactionIntelligence } from '@/features/transaction-intelligence/TransactionIntelligence';
import { TransactionTable } from '@/features/transactions';
import { useDashboardSummary } from '@/lib/queries';
import { fmt, human } from '@/lib/format';
export default function OverviewPage() {
  const query = useDashboardSummary();
  return (
    <>
      <PageHeader
        eyebrow="ANALYST WORKSPACE"
        title="Risk overview"
        description="Observed transactions, evidence and review activity. A clear view of what needs attention."
        actions={
          <Link className="button" to="/app/transactions">
            <ListFilter size={15} />
            Explore transactions
          </Link>
        }
      />
      <div className="page-stack">
        <DataState query={query}>
          {(data) => (
            <>
              <div className="metric-grid">
                <MetricDisplay
                  label="Observed transactions"
                  value={fmt(data.transaction_count, 0)}
                  note="Current demo activity"
                  icon={<Activity size={16} />}
                />
                <MetricDisplay
                  label="Investigation flags"
                  value={fmt(data.investigation_count, 0)}
                  note="Frozen ABRF decision policy"
                  icon={<ShieldCheck size={16} />}
                />
                <MetricDisplay
                  label="Average fused risk"
                  value={fmt(data.average_risk_score, 4)}
                  note="Score, not fraud prevalence"
                  icon={<SlidersHorizontal size={16} />}
                />
                <MetricDisplay
                  label="Elapsed-day buckets"
                  value={fmt(data.risk_trend.length, 0)}
                  note="Synthetic relative time"
                />
              </div>
              <div className="grid-2">
                <ChartContainer
                  title="Risk over relative time"
                  note="Average fused risk across synthetic elapsed days. Not calendar dates."
                  data={data.risk_trend}
                  x="elapsed_day_index"
                  kind="area"
                  series={[{ key: 'average_risk_score', label: 'Average risk' }]}
                />
                <ChartContainer
                  title="Risk distribution"
                  note="Risk bands are independent of the binary investigation flag."
                  data={Object.entries(data.risk_level_distribution).map(([name, count]) => ({
                    name: human(name),
                    count,
                  }))}
                  x="name"
                  series={[{ key: 'count', label: 'Transactions' }]}
                />
              </div>
              <Panel
                title="Recent investigation flags"
                note="Actual saved predictions; analyst decisions do not change model scores."
              >
                <TransactionTable items={data.recent_high_risk_transactions} details={false} />
              </Panel>
              <TransactionIntelligence transactions={data.recent_transaction_activity} />
              <ChartContainer
                title="Transaction activity"
                note={data.scope}
                data={data.risk_trend}
                x="elapsed_day_index"
                series={[{ key: 'transactions', label: 'Observed transactions' }]}
              />
            </>
          )}
        </DataState>
        <SectionHeader
          title="Frozen final evaluation"
          note="Separate from current dashboard activity. Synthetic future holdout, evaluated once."
          action={
            <Link to="/app/performance" className="text-link">
              Performance report <ArrowUpRight size={14} />
            </Link>
          }
        />
        <FinalMetrics />
      </div>
    </>
  );
}
