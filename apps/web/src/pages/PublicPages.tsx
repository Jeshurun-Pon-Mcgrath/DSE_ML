import { ArrowRight, ArrowUpRight, CheckCircle2, Eye, Layers, ShieldCheck } from 'lucide-react';
import { Link, Outlet } from 'react-router-dom';
import { Brand, DataState, DefinitionList, Notice, PageHeader, Panel } from '@/components/ui';
import { FinalMetrics, ScenarioPerformance } from '@/features/performance';
import { useModelInfo } from '@/lib/queries';
import { ResearchData } from '@/features/research';
import { number, text, valueAt } from '@/lib/api';
import { fmt } from '@/lib/format';
export function PublicLayout() {
  return (
    <div className="public-page">
      <a className="skip-link" href="#main-content">
        Skip to content
      </a>
      <header className="public-header">
        <Brand />
        <nav className="public-nav" aria-label="Public navigation">
          <Link to="/technology">How it works</Link>
          <Link to="/app/performance">Evaluation</Link>
          <Link className="button" to="/app">
            Open workspace <ArrowUpRight size={15} />
          </Link>
        </nav>
      </header>
      <main id="main-content" className="public-main">
        <Outlet />
      </main>
      <footer className="public-footer">
        <span>RiskByte · Explainable Financial Fraud Intelligence</span>
        <nav aria-label="Legal navigation">
          <Link to="/technology">Technology</Link>
          <Link to="/privacy">Privacy</Link>
          <Link to="/terms">Terms</Link>
        </nav>
      </footer>
    </div>
  );
}
export function HomePage() {
  const model = useModelInfo();
  return (
    <>
      <section className="hero">
        <div>
          <div className="eyebrow">EXPLAINABLE FINANCIAL FRAUD INTELLIGENCE</div>
          <h1>
            Detect suspicious
            <br />
            transactions.<span className="sub-line">Understand why.</span>
          </h1>
          <p>
            Investigate risk with evidence. RiskByte connects supervised classification, prior
            behavioural history and transparent risk fusion in one analyst workspace.
          </p>
          <div className="hero-actions">
            <Link className="button" to="/app">
              Open analyst workspace <ArrowRight size={15} />
            </Link>
            <Link className="button secondary" to="/technology">
              Explore the methodology
            </Link>
          </div>
          <div className="legend-lines">
            <span>
              <CheckCircle2 size={12} className="inline" /> Frozen evaluation
            </span>
            <span>Synthetic academic benchmark</span>
          </div>
        </div>
        <div className="hero-architecture">
          <div className="eyebrow">THE INTELLIGENCE STACK</div>
          <div className="architecture-step">
            <strong>Transaction context</strong>
            <span>STRICT VALIDATION</span>
          </div>
          <div className="architecture-step">
            <strong>Decision Tree + calibration</strong>
            <span>SUPERVISED P</span>
          </div>
          <div className="architecture-step">
            <strong>Strictly prior behaviour</strong>
            <span>DEVIATION B + CONFIDENCE</span>
          </div>
          <div className="architecture-step">
            <strong>Adaptive risk fusion</strong>
            <span>ABRF</span>
          </div>
          <div className="architecture-step">
            <strong>Evidence-led investigation</strong>
            <span>SHAP + CALCULATION TRACE</span>
          </div>
          <p className="architecture-caption">
            Isolation Forest is retained diagnostically. It has zero influence in the frozen active
            fusion. No risk score establishes fraud guilt.
          </p>
        </div>
      </section>
      <section className="public-section">
        <PageHeader
          eyebrow="MEASURED, NOT ASSUMED"
          level={2}
          title="A frozen benchmark. Honest results."
          description="Final performance comes from the backend’s locked future holdout. The benchmark contains synthetic academic transactions, not real banking customers."
        />
        <FinalMetrics />
      </section>
      <section className="public-section">
        <PageHeader
          eyebrow="FOUR LAYERS OF EVIDENCE"
          level={2}
          title="A flag is the beginning of an investigation."
          description="Inspect what the classifier used, what differed from prior activity and how the final decision-support score was calculated."
        />
        <div className="grid-3">
          {[
            [
              ShieldCheck,
              'Classifier explanation',
              'Genuine Tree SHAP attributes the underlying Decision Tree output. Calibration is shown separately.',
            ],
            [
              Eye,
              'Behavioural evidence',
              'Amount deviation, velocity and novelty use only the entity’s strictly prior transaction history.',
            ],
            [
              Layers,
              'Transparent fusion',
              'Base weights, confidence-adjusted weights and contributions reproduce the final fused risk exactly.',
            ],
          ].map(([Icon, title, description]) => {
            const Glyph = typeof Icon === 'string' ? ShieldCheck : (Icon ?? ShieldCheck);
            return (
              <Panel key={String(title)}>
                <Glyph size={25} color="var(--teal)" />
                <h3 className="mt-5 mb-3">{String(title)}</h3>
                <p className="caption">{String(description)}</p>
              </Panel>
            );
          })}
        </div>
      </section>
      <section className="public-section">
        <PageHeader
          eyebrow="SCENARIO COVERAGE"
          level={2}
          title="Strengths and blind spots, in view."
          description="Account takeover and mixed-mode attacks are strong cases. Velocity bursts remain a limitation. Results are never hidden behind accuracy alone."
        />
        <ScenarioPerformance />
      </section>
      <section className="public-section">
        <DataState query={model}>{(m) => <Notice>{m.notice}</Notice>}</DataState>
        <div className="hero-actions">
          <Link className="button" to="/app">
            Investigate the evidence <ArrowRight size={15} />
          </Link>
        </div>
      </section>
    </>
  );
}
export function TechnologyPage() {
  return (
    <>
      <div className="prose">
        <PageHeader
          eyebrow="SYSTEM METHODOLOGY"
          title="From transaction to evidence."
          description="Three distinct intelligence signals. One reproducible, inspectable decision-support calculation."
        />
        <h2>Supervised classification</h2>
        <p>
          The frozen Decision Tree learns known synthetic fraud patterns. A sigmoid calibration
          layer produces the supervised probability P. The final operating policy is preserved; no
          model is trained in the browser.
        </p>
        <h2>Behavioural deviation</h2>
        <p>
          Each transaction is compared with its synthetic entity’s strictly prior history. Amount
          deviation, transaction velocity and novelty remain visible. Confidence decreases when
          history or component availability is limited. A new entity is uncertain, not automatically
          safe.
        </p>
        <h2>Anomaly diagnostics</h2>
        <p>
          Isolation Forest identifies unusual observations using its frozen training distribution.
          Its normalized score A is not a fraud probability. Validation selected zero anomaly weight
          in the active ABRF configuration, so diagnostic anomaly evidence does not raise the final
          score.
        </p>
        <h2>Confidence-aware fusion</h2>
        <div className="formula">R = (wₚP + wₐA + wᵦg(Cᵦ)B) / (wₚ + wₐ + wᵦg(Cᵦ))</div>
        <ResearchData path="/abrf/config">
          {(d) => (
            <DefinitionList
              items={[
                [
                  'Supervised base weight',
                  fmt(number(valueAt(d.configuration, 'weights', 'wp')), 1),
                ],
                ['Anomaly base weight', fmt(number(valueAt(d.configuration, 'weights', 'wa')), 1)],
                [
                  'Behaviour base weight',
                  fmt(number(valueAt(d.configuration, 'weights', 'wb')), 1),
                ],
                [
                  'Investigation threshold',
                  number(valueAt(d.configuration, 'decision_threshold'))?.toString() ??
                    'Unavailable',
                ],
                ['ABRF version', text(valueAt(d.configuration, 'version'))],
              ]}
            />
          )}
        </ResearchData>
        <h2>Explanation scope</h2>
        <p>
          Tree SHAP explains feature contributions to the underlying Decision Tree output. It does
          not include sigmoid calibration or explain the entire fusion. Behavioural evidence and the
          ABRF trace are direct calculations. SHAP does not establish causation, criminal intent,
          fraud guilt or human motivation.
        </p>
        <h2>Reproducibility and persistence</h2>
        <p>
          The interface uses versioned FastAPI endpoints. Frozen dataset fingerprints, feature
          ordering, model artifacts and risk configuration are verified. PostgreSQL-ready SQLAlchemy
          persistence stores transactions, predictions, analyst decisions and safe audit events; the
          local academic demo uses SQLite.
        </p>
        <Link className="button" to="/app/system">
          Inspect system provenance
        </Link>
      </div>
    </>
  );
}
export function PrivacyPage() {
  return (
    <article className="prose">
      <PageHeader
        eyebrow="POLICY / ACADEMIC DEMO"
        title="Privacy policy"
        description="What this demonstration handles and what it does not claim."
      />
      <h2>Synthetic data</h2>
      <p>
        The provided benchmark and demo records are synthetic financial transactions. They are not
        original IEEE-CIS competition records or records of real banking customers. Do not submit
        personal, confidential or real financial information to this academic environment.
      </p>
      <h2>Submitted transaction context</h2>
      <p>
        Prediction requests can include transaction amounts, synthetic entity keys and device,
        merchant or region context. The backend stores submitted context, computed predictions and
        analyst review states. This helps preserve strictly prior history and reproducible
        explanations.
      </p>
      <h2>Operational records</h2>
      <p>
        Audit events record timestamps, transaction identifiers and safe action metadata. Model
        scores remain unchanged when a review state changes. No public endpoint should expose
        credentials or local filesystem paths.
      </p>
      <h2>Browser storage and access</h2>
      <p>
        Local storage retains display density and table-size preferences. The interface does not
        provide enterprise authentication or legal compliance certification. Operators control
        hosting, access, database backups, retention and deletion. Production write credentials
        belong in a trusted server-side integration, never public frontend configuration.
      </p>
      <h2>Requests and responsibility</h2>
      <p>
        For questions about a deployed instance, contact that instance’s operator through their
        published channel. This project does not claim a statutory retention schedule or verified
        regulatory certification.
      </p>
    </article>
  );
}
export function TermsPage() {
  return (
    <article className="prose">
      <PageHeader
        eyebrow="TERMS / RESEARCH DEMONSTRATION"
        title="Terms and conditions"
        description="The scope and appropriate use of the RiskByte academic platform."
      />
      <h2>Research use</h2>
      <p>
        RiskByte is an academic research and demonstration system. All reported evaluation results
        describe a synthetic benchmark. They cannot establish performance in production banking or
        another financial population.
      </p>
      <h2>Decision support</h2>
      <p>
        Scores and investigation flags recommend human review. They are not a financial decision
        authority, proof of fraud, a finding of guilt or a determination of human intent. Review
        evidence and limitations before drawing conclusions.
      </p>
      <h2>Explanation limitations</h2>
      <p>
        Classifier SHAP describes model output, not causation. Behaviour and anomaly scores describe
        deviations, not fraud probabilities. Active ABRF is a project-specific methodology, not a
        universally optimal mathematical algorithm.
      </p>
      <h2>Responsible operation</h2>
      <p>
        Use only authorized synthetic inputs. Do not submit secrets or real customer records,
        attempt unauthorized access or treat this demo as an approved live banking service.
        Operators must establish appropriate security and access controls before exposing an
        instance.
      </p>
      <h2>Frozen evaluation</h2>
      <p>
        The final holdout was evaluated once. Modifying a model, threshold or generating rule
        invalidates claims based on the frozen final snapshot. Analyst workflow changes do not
        modify the snapshot.
      </p>
    </article>
  );
}
export function NotFoundPage() {
  return (
    <div className="prose">
      <div className="error-code">404</div>
      <PageHeader
        title="Page not found"
        description="This route is not part of the RiskByte workspace. Return to the overview or transaction explorer."
      />
      <div className="hero-actions">
        <Link className="button" to="/app">
          Open workspace
        </Link>
        <Link className="button secondary" to="/">
          Home
        </Link>
      </div>
    </div>
  );
}
