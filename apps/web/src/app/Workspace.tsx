import { useEffect, useRef, useState } from 'react';
import {
  Activity,
  ArrowUpRight,
  BarChart3,
  BookOpen,
  ChevronRight,
  Database,
  FileClock,
  FlaskConical,
  GitMerge,
  LayoutDashboard,
  Menu,
  ScanLine,
  Search,
  Settings,
  ShieldCheck,
  SlidersHorizontal,
  X,
} from 'lucide-react';
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom';
import { Brand, Breadcrumb, StatusIndicator } from '@/components/ui';
import { useHealth } from '@/lib/queries';
const groups = [
  {
    label: 'Workspace',
    items: [
      { path: '/app', label: 'Overview', icon: LayoutDashboard },
      { path: '/app/transactions', label: 'Transactions', icon: Search },
      { path: '/app/investigations', label: 'Investigation queue', icon: ShieldCheck },
    ],
  },
  {
    label: 'Intelligence',
    items: [
      { path: '/app/explainability', label: 'Risk explanation', icon: ScanLine },
      { path: '/app/behaviour', label: 'Behaviour intelligence', icon: Activity },
      { path: '/app/anomaly', label: 'Anomaly intelligence', icon: SlidersHorizontal },
      { path: '/app/abrf', label: 'ABRF fusion', icon: GitMerge },
    ],
  },
  {
    label: 'Research & assurance',
    items: [
      { path: '/app/dataset', label: 'Dataset & EDA', icon: Database },
      { path: '/app/statistics', label: 'Statistical analysis', icon: FlaskConical },
      { path: '/app/performance', label: 'Model performance', icon: BarChart3 },
      { path: '/app/audit', label: 'Audit history', icon: FileClock },
      { path: '/app/system', label: 'System & model', icon: BookOpen },
      { path: '/app/settings', label: 'Settings', icon: Settings },
    ],
  },
];
export default function Workspace() {
  const [open, setOpen] = useState(false);
  const health = useHealth();
  const location = useLocation();
  const ref = useRef<HTMLElement>(null);
  const current =
    groups
      .flatMap((g) => g.items)
      .filter(
        (i) =>
          location.pathname === i.path ||
          (i.path !== '/app' && location.pathname.startsWith(`${i.path}/`)),
      )
      .at(-1)?.label ?? 'Workspace';
  useEffect(() => {
    if (!open) return;
    const previous = document.activeElement;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    const focusable = () =>
      Array.from(ref.current?.querySelectorAll<HTMLElement>('a,button') ?? []);
    focusable()[0]?.focus();
    function keyboard(e: KeyboardEvent) {
      if (e.key === 'Escape') setOpen(false);
      if (e.key === 'Tab') {
        const nodes = focusable(),
          first = nodes[0],
          last = nodes.at(-1);
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last?.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first?.focus();
        }
      }
    }
    document.addEventListener('keydown', keyboard);
    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener('keydown', keyboard);
      if (previous instanceof HTMLElement) previous.focus();
    };
  }, [open]);
  return (
    <div className="workspace">
      <a className="skip-link" href="#main-content">
        Skip to content
      </a>
      {open && (
        <button
          className="nav-scrim"
          aria-label="Close navigation"
          tabIndex={-1}
          onClick={() => setOpen(false)}
        />
      )}
      <aside
        ref={ref}
        className={`sidebar ${open ? 'open' : ''}`}
        {...(open
          ? { role: 'dialog', 'aria-modal': true, 'aria-label': 'Workspace navigation' }
          : {})}
      >
        <Brand />
        <button
          className="icon-button mobile-close"
          aria-label="Close navigation"
          onClick={() => setOpen(false)}
        >
          <X size={17} />
        </button>
        <nav className="sidebar-nav" aria-label="Analyst navigation">
          {groups.map((g) => (
            <div key={g.label}>
              <div className="nav-group">{g.label}</div>
              {g.items.map((i) => (
                <NavLink
                  end={i.path === '/app'}
                  className="nav-link"
                  key={i.path}
                  to={i.path}
                  onClick={() => setOpen(false)}
                >
                  <i.icon size={16} />
                  {i.label}
                </NavLink>
              ))}
            </div>
          ))}
        </nav>
        <div className="sidebar-footer">
          Academic synthetic environment
          <span>
            Evidence supports review.
            <br />A risk flag does not establish guilt.
          </span>
          <Link className="text-link mt-3" to="/technology">
            Methodology <ArrowUpRight size={12} />
          </Link>
        </div>
      </aside>
      <div className="workspace-main">
        <header className="topbar">
          <div className="flex items-center gap-3">
            <button
              className="icon-button mobile-menu"
              aria-label="Open navigation"
              aria-expanded={open}
              onClick={() => setOpen(true)}
            >
              <Menu size={18} />
            </button>
            <Breadcrumb current={current} />
          </div>
          <div className="topbar-right">
            <span className="caption">SYNTHETIC RESEARCH WORKSPACE</span>
            {health.isPending ? (
              <span className="caption">Checking system</span>
            ) : (
              <StatusIndicator
                healthy={!health.isError && health.data?.inference_ready === true}
                label={
                  !health.isError && health.data?.inference_ready
                    ? 'System ready'
                    : 'System unavailable'
                }
              />
            )}
            <Link to="/app/settings" aria-label="Workspace settings">
              <Settings size={16} />
            </Link>
          </div>
        </header>
        <main id="main-content" className="workspace-content">
          <Outlet />
          <footer className="workspace-footer">
            <span>RiskByte · Explainable Financial Fraud Intelligence</span>
            <Link to="/app/system" className="text-link">
              Frozen system provenance <ChevronRight size={12} />
            </Link>
          </footer>
        </main>
      </div>
    </div>
  );
}
