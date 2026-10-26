import {
  lazy,
  Suspense,
  useEffect,
  useRef,
  type ButtonHTMLAttributes,
  type ReactNode,
} from 'react';
import {
  ArrowLeft,
  ArrowRight,
  CheckCircle2,
  ChevronRight,
  FileSearch,
  RefreshCw,
  ShieldCheck,
  TriangleAlert,
  X,
} from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import { fmt } from '@/lib/format';
import type { ChartProps } from './Chart';
const Chart = lazy(() => import('./Chart'));
export function Brand() {
  return (
    <Link to="/" className="brand" aria-label="RiskByte home">
      <span className="brand-symbol">
        <ShieldCheck size={23} />
      </span>
      <span>
        Risk<span className="brand-byte">Byte</span>
        <small>FINANCIAL INTELLIGENCE</small>
      </span>
    </Link>
  );
}
export function Button({
  children,
  className = '',
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button className={`button ${className}`} {...props}>
      {children}
    </button>
  );
}
export function IconButton({
  label,
  children,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { label: string }) {
  return (
    <button className="icon-button" aria-label={label} title={label} {...props}>
      {children}
    </button>
  );
}
export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
  level = 1,
}: {
  eyebrow?: string;
  title: string;
  description: string;
  actions?: ReactNode;
  level?: 1 | 2;
}) {
  const Heading = level === 1 ? 'h1' : 'h2';
  return (
    <header className="page-header">
      <div>
        {eyebrow && <div className="eyebrow">{eyebrow}</div>}
        <Heading>{title}</Heading>
        <p>{description}</p>
      </div>
      {actions && <div className="header-actions">{actions}</div>}
    </header>
  );
}
export function SectionHeader({
  title,
  note,
  action,
}: {
  title: string;
  note?: string;
  action?: ReactNode;
}) {
  return (
    <div className="section-header">
      <div>
        <h2>{title}</h2>
        {note && <p>{note}</p>}
      </div>
      {action}
    </div>
  );
}
export function Panel({
  children,
  className = '',
  title,
  note,
}: {
  children: ReactNode;
  className?: string;
  title?: string;
  note?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      {title && <SectionHeader title={title} {...(note ? { note } : {})} />}
      <div className="panel-body">{children}</div>
    </section>
  );
}
export function MetricDisplay({
  label,
  value,
  note,
  icon,
}: {
  label: string;
  value: string;
  note?: string;
  icon?: ReactNode;
}) {
  return (
    <article className="metric">
      <div className="metric-label">
        {label}
        {icon}
      </div>
      <strong>{value}</strong>
      {note && <span className="caption">{note}</span>}
    </article>
  );
}
export function RiskBadge({ level }: { level: string }) {
  return (
    <span className={`risk-badge risk-${level.toLowerCase()}`}>
      <span className="risk-dot" />
      {level}
    </span>
  );
}
export function StatusIndicator({ healthy, label }: { healthy: boolean; label: string }) {
  return (
    <span className={`status ${healthy ? 'healthy' : 'degraded'}`}>
      {healthy ? <CheckCircle2 size={14} /> : <TriangleAlert size={14} />} {label}
    </span>
  );
}
export function RiskMeter({ score, level }: { score: number; level: string }) {
  return (
    <div className="risk-meter">
      <div className="risk-meter-heading">
        <span className="eyebrow">FUSED RISK SCORE</span>
        <RiskBadge level={level} />
      </div>
      <div className="risk-number">
        {fmt(score, 4)}
        <span>/ 1.00</span>
      </div>
      <div
        className="meter-track"
        role="meter"
        aria-label="Fused risk score"
        aria-valuemin={0}
        aria-valuemax={1}
        aria-valuenow={score}
      >
        <div
          style={{ width: `${score * 100}%` }}
          className={`meter-fill risk-${level.toLowerCase()}`}
        />
      </div>
      <p className="caption">Decision-support score. Not a fraud probability.</p>
    </div>
  );
}
export function EmptyState({
  title = 'No records found',
  description = 'Try a different filter or search.',
}: {
  title?: string;
  description?: string;
}) {
  return (
    <div className="empty-state">
      <FileSearch size={28} />
      <h3>{title}</h3>
      <p>{description}</p>
    </div>
  );
}
export function ErrorState({ error, retry }: { error: Error | null; retry: () => void }) {
  return (
    <div className="error-state" role="alert">
      <TriangleAlert size={24} />
      <h3>Data could not be loaded</h3>
      <p>{error?.message ?? 'The service is unavailable.'}</p>
      <Button onClick={retry}>
        <RefreshCw size={15} />
        Try again
      </Button>
    </div>
  );
}
export function Skeleton() {
  return (
    <div className="skeleton-block" role="status" aria-label="Loading data">
      <span className="skeleton-line" />
      <span className="skeleton-line" />
      <span className="skeleton-line" />
      <span className="sr-only">Loading data</span>
    </div>
  );
}
export function DataState<T>({
  query,
  children,
}: {
  query: {
    isPending: boolean;
    isError: boolean;
    error: Error | null;
    data: T | undefined;
    refetch: () => unknown;
  };
  children: (data: T) => ReactNode;
}) {
  if (query.isPending) return <Skeleton />;
  if (query.isError)
    return (
      <ErrorState
        error={query.error}
        retry={() => {
          query.refetch();
        }}
      />
    );
  if (query.data === undefined)
    return (
      <EmptyState
        title="Data unavailable"
        description="This dataset or analysis has not been configured."
      />
    );
  return <>{children(query.data)}</>;
}
export function ChartContainer(props: ChartProps & { title: string; note?: string }) {
  return (
    <Panel title={props.title} {...(props.note ? { note: props.note } : {})}>
      <Suspense fallback={<Skeleton />}>
        <Chart {...props} />
      </Suspense>
    </Panel>
  );
}
export function Pagination({
  page,
  total,
  pageSize,
  onChange,
}: {
  page: number;
  total: number;
  pageSize: number;
  onChange: (page: number) => void;
}) {
  const last = Math.max(1, Math.ceil(total / pageSize));
  return (
    <nav className="pagination" aria-label="Table pagination">
      <span className="caption">
        {fmt(total, 0)} records · Page {page} of {last}
      </span>
      <div>
        <IconButton label="Previous page" disabled={page <= 1} onClick={() => onChange(page - 1)}>
          <ArrowLeft size={16} />
        </IconButton>
        <IconButton label="Next page" disabled={page >= last} onClick={() => onChange(page + 1)}>
          <ArrowRight size={16} />
        </IconButton>
      </div>
    </nav>
  );
}
export function Dialog({
  open,
  onClose,
  title,
  children,
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    if (open && !ref.current?.open) ref.current?.showModal();
    else if (!open && ref.current?.open) ref.current.close();
  }, [open]);
  return (
    <dialog ref={ref} className="dialog" aria-labelledby="dialog-title" onCancel={onClose}>
      <div className="section-header">
        <h2 id="dialog-title">{title}</h2>
        <IconButton label="Close dialog" onClick={onClose}>
          <X size={20} />
        </IconButton>
      </div>
      <div className="panel-body">{children}</div>
    </dialog>
  );
}
export function Breadcrumb({ current }: { current: string }) {
  return (
    <nav aria-label="Breadcrumb" className="breadcrumb">
      <Link to="/app">Workspace</Link>
      <ChevronRight size={13} />
      <span aria-current="page">{current}</span>
    </nav>
  );
}
export function DataTable({
  headers,
  rows,
  label,
}: {
  headers: string[];
  rows: { id: string; cells: ReactNode[]; href?: string }[];
  label: string;
}) {
  const navigate = useNavigate();
  if (!rows.length) return <EmptyState />;
  return (
    <div className="table-container">
      <table className="data-table" aria-label={label}>
        <thead>
          <tr>
            {headers.map((h) => (
              <th key={h} scope="col">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr
              key={r.id}
              className={r.href ? 'clickable-row' : undefined}
              onClick={(event) => {
                if (
                  r.href &&
                  event.target instanceof Element &&
                  !event.target.closest('a,button,input,select')
                )
                  void navigate(r.href);
              }}
            >
              {r.cells.map((c, i) => (
                <td key={headers[i] ?? i} data-label={headers[i]}>
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
export function DefinitionList({ items }: { items: [string, ReactNode][] }) {
  return (
    <dl className="definition-list">
      {items.map(([key, value]) => (
        <div key={key}>
          <dt>{key}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  );
}
export function Notice({ children }: { children: ReactNode }) {
  return (
    <div className="notice">
      <ShieldCheck size={18} />
      <p>{children}</p>
    </div>
  );
}
export function Toast({ children }: { children: ReactNode }) {
  return (
    <div className="toast" role="status">
      <CheckCircle2 size={18} />
      {children}
    </div>
  );
}
