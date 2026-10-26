import { Component, lazy, Suspense, useEffect, type ReactNode } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter, Route, Routes, useLocation } from 'react-router-dom';
import { Button, Skeleton } from '@/components/ui';
const Workspace = lazy(() => import('./Workspace'));
const Overview = lazy(() => import('@/pages/OverviewPage'));
const Transactions = lazy(() => import('@/pages/TransactionsPage'));
const Transaction = lazy(() => import('@/pages/TransactionPage'));
const Intelligence = lazy(() => import('@/pages/IntelligencePages'));
const Dataset = lazy(() =>
  import('@/pages/ResearchPages').then((m) => ({ default: m.DatasetPage })),
);
const Statistics = lazy(() =>
  import('@/pages/ResearchPages').then((m) => ({ default: m.StatisticsPage })),
);
const Performance = lazy(() =>
  import('@/pages/ResearchPages').then((m) => ({ default: m.PerformancePage })),
);
const System = lazy(() => import('@/pages/SystemPages').then((m) => ({ default: m.SystemPage })));
const Audit = lazy(() => import('@/pages/SystemPages').then((m) => ({ default: m.AuditPage })));
const Settings = lazy(() =>
  import('@/pages/SystemPages').then((m) => ({ default: m.SettingsPage })),
);
const Public = lazy(() => import('@/pages/PublicPages').then((m) => ({ default: m.PublicLayout })));
const Home = lazy(() => import('@/pages/PublicPages').then((m) => ({ default: m.HomePage })));
const Technology = lazy(() =>
  import('@/pages/PublicPages').then((m) => ({ default: m.TechnologyPage })),
);
const Privacy = lazy(() => import('@/pages/PublicPages').then((m) => ({ default: m.PrivacyPage })));
const Terms = lazy(() => import('@/pages/PublicPages').then((m) => ({ default: m.TermsPage })));
const NotFound = lazy(() =>
  import('@/pages/PublicPages').then((m) => ({ default: m.NotFoundPage })),
);
const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 30000, retry: 1, refetchOnWindowFocus: false },
    mutations: { retry: false },
  },
});
class ErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? (
      <main className="error-state">
        <h1>The workspace could not open</h1>
        <p>Refresh to reload the application. If this continues, contact the operator.</p>
        <Button onClick={() => window.location.reload()}>Reload workspace</Button>
      </main>
    ) : (
      this.props.children
    );
  }
}
function RouteMetadata() {
  const { pathname } = useLocation();
  useEffect(() => {
    const key = pathname.split('/').filter(Boolean).at(-1) ?? 'Home';
    const title =
      key === 'app'
        ? 'Overview'
        : key.replaceAll('-', ' ').replace(/\b\w/g, (c) => c.toUpperCase());
    document.title = `${title} | RiskByte`;
    const description =
      pathname === '/'
        ? 'Detect suspicious transactions. Understand why they were flagged. Investigate risk with evidence.'
        : `${title}: inspect RiskByte synthetic financial transaction intelligence, evidence and frozen system results.`;
    document.querySelector('meta[name="description"]')?.setAttribute('content', description);
    document.querySelector('meta[property="og:title"]')?.setAttribute('content', document.title);
    document.querySelector('meta[property="og:description"]')?.setAttribute('content', description);
    document.documentElement.dataset.density =
      localStorage.getItem('riskbyte.compact') === 'true' ? 'compact' : 'comfortable';
    window.scrollTo({ top: 0, behavior: 'instant' });
  }, [pathname]);
  return null;
}
export function App() {
  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <RouteMetadata />
          <Suspense fallback={<Skeleton />}>
            <Routes>
              <Route element={<Public />}>
                <Route index element={<Home />} />
                <Route path="technology" element={<Technology />} />
                <Route path="privacy" element={<Privacy />} />
                <Route path="terms" element={<Terms />} />
                <Route path="*" element={<NotFound />} />
              </Route>
              <Route path="app" element={<Workspace />}>
                <Route index element={<Overview />} />
                <Route path="transactions" element={<Transactions />} />
                <Route path="transactions/:id" element={<Transaction key="detail" />} />
                <Route
                  path="transactions/:id/explanation"
                  element={<Transaction key="explain" explain />}
                />
                <Route path="investigations" element={<Transactions queue />} />
                <Route path="behaviour" element={<Intelligence kind="behaviour" />} />
                <Route path="anomaly" element={<Intelligence kind="anomaly" />} />
                <Route path="abrf" element={<Intelligence kind="abrf" />} />
                <Route path="explainability" element={<Intelligence kind="explainability" />} />
                <Route path="dataset" element={<Dataset />} />
                <Route path="statistics" element={<Statistics />} />
                <Route path="performance" element={<Performance />} />
                <Route path="performance/scenarios" element={<Performance view="scenarios" />} />
                <Route
                  path="performance/hard-negatives"
                  element={<Performance view="hard-negatives" />}
                />
                <Route path="audit" element={<Audit />} />
                <Route path="system" element={<System />} />
                <Route path="settings" element={<Settings />} />
                <Route path="*" element={<NotFound />} />
              </Route>
            </Routes>
          </Suspense>
        </BrowserRouter>
      </QueryClientProvider>
    </ErrorBoundary>
  );
}
