import { lazy, Suspense, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Plus, Search } from 'lucide-react';
import {
  Button,
  DataState,
  Dialog,
  PageHeader,
  Pagination,
  Panel,
  Skeleton,
} from '@/components/ui';
import { levels, states } from '@/lib/api';
import { useTransactions } from '@/lib/queries';
import { human } from '@/lib/format';
import { TransactionTable } from '@/features/transactions';
const PredictionForm = lazy(() => import('@/features/prediction-form'));
export default function TransactionsPage({ queue = false }: { queue?: boolean }) {
  const [params, setParams] = useSearchParams();
  const [open, setOpen] = useState(false);
  const page = Math.max(1, Number(params.get('page')) || 1);
  const pageSize = Number(localStorage.getItem('riskbyte.pageSize') ?? '15');
  const filters = {
    page: String(page),
    page_size: String([15, 25, 50].includes(pageSize) ? pageSize : 15),
    sort: params.get('sort') ?? 'recent',
    ...(params.get('search') ? { search: params.get('search') ?? '' } : {}),
    ...(params.get('risk_level') ? { risk_level: params.get('risk_level') ?? '' } : {}),
    ...(params.get('investigation_state')
      ? { investigation_state: params.get('investigation_state') ?? '' }
      : {}),
    ...(queue ? { investigation_required: 'true' } : {}),
  };
  const query = useTransactions(filters);
  function change(key: string, value: string) {
    const next = new URLSearchParams(window.location.search);
    if (value) next.set(key, value);
    else next.delete(key);
    if (key !== 'page') next.delete('page');
    setParams(next);
  }
  return (
    <>
      <PageHeader
        eyebrow={queue ? 'ANALYST REVIEW' : 'TRANSACTION EXPLORER'}
        title={queue ? 'Investigation queue' : 'Transactions'}
        description={
          queue
            ? 'Review flagged transactions, record decisions and preserve the original model evidence.'
            : 'Inspect observed transactions and their frozen intelligence outputs.'
        }
        actions={
          !queue ? (
            <Button onClick={() => setOpen(true)}>
              <Plus size={15} />
              New prediction
            </Button>
          ) : (
            <span className="surface-label">Scores remain immutable</span>
          )
        }
      />
      <div className="filters">
        <div className="field">
          <label htmlFor="search">
            <Search size={12} className="inline" /> Search transaction ID
          </label>
          <input
            id="search"
            value={params.get('search') ?? ''}
            maxLength={100}
            onChange={(e) => change('search', e.target.value)}
            type="search"
          />
        </div>
        <div className="field">
          <label htmlFor="risk">Risk level</label>
          <select
            id="risk"
            value={params.get('risk_level') ?? ''}
            onChange={(e) => change('risk_level', e.target.value)}
          >
            <option value="">All levels</option>
            {levels.map((level) => (
              <option key={level}>{level}</option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor="review">Review state</label>
          <select
            id="review"
            value={params.get('investigation_state') ?? ''}
            onChange={(e) => change('investigation_state', e.target.value)}
          >
            <option value="">All states</option>
            {states.map((state) => (
              <option value={state} key={state}>
                {human(state)}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor="sort">Sort by</label>
          <select
            id="sort"
            value={params.get('sort') ?? 'recent'}
            onChange={(e) => change('sort', e.target.value)}
          >
            <option value="recent">Most recent</option>
            <option value="risk">Highest risk</option>
          </select>
        </div>
        <Button className="secondary" onClick={() => setParams({})}>
          Reset
        </Button>
      </div>
      <DataState query={query}>
        {(data) => (
          <Panel>
            <TransactionTable items={data.items} />
            <Pagination
              page={page}
              total={data.total}
              pageSize={[15, 25, 50].includes(pageSize) ? pageSize : 15}
              onChange={(p) => change('page', String(p))}
            />
            {query.isFetching && (
              <p className="caption" role="status">
                Updating results…
              </p>
            )}
          </Panel>
        )}
      </DataState>
      <Dialog open={open} onClose={() => setOpen(false)} title="Evaluate a synthetic transaction">
        <Suspense fallback={<Skeleton />}>
          {open && <PredictionForm onClose={() => setOpen(false)} />}
        </Suspense>
      </Dialog>
    </>
  );
}
