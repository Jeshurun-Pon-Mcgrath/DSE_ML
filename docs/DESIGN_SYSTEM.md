# RiskByte design system

## Active final product system

The product uses a restrained analyst-workstation design: graphite navigation, warm
off-white surfaces, steel typography and controlled teal emphasis. Styles and Tailwind
theme integration live in `apps/web/src/styles/index.css`. No stock imagery, gradients,
glass effects, custom cursors or decorative animation is used.

| Token family      | Convention                                                            |
| ----------------- | --------------------------------------------------------------------- |
| Canvas/surface    | #f4f3ee / #fffefa                                                     |
| Graphite/ink      | #131f25 / #182329                                                     |
| Analytical accent | teal #12696a, cyan #69b9bb on dark surfaces                           |
| Risk              | LOW green, GUARDED steel, ELEVATED amber, HIGH orange, CRITICAL red   |
| Spacing           | 4px base; explicit 4/8/12/16/20/24/32/36/48px tokens                  |
| Type              | Local Segoe UI/system sans; Cascadia Code/Consolas for technical data |
| Type roles        | display, page, section, body, caption, data label and metric tokens   |
| Radius            | 4px panels/controls; 6px native dialogs; restrained borders           |
| Elevation         | subtle panel shadow; separation primarily through borders             |
| Motion            | 180ms control/navigation feedback; no chart entrance animation        |
| Chart             | teal/steel/amber series, explicit legends and formatted tooltips      |

Green is reserved for verified healthy states, successful operations and LOW risk bands.
Decreasing SHAP attribution uses neutral teal; increasing attribution uses amber. Feature
attribution does not establish transaction legitimacy or causation. Risk colours always
appear with text, not as the only communication channel.

Visible focus, semantic labels, one primary heading, native modal focus handling,
keyboard/Escape mobile navigation, loading/error/empty states and reduced-motion support
are implemented. Mobile tables become labelled records; the shared layout was checked
at 1920, 1440, 1280, 1024, 768 and 430px. Display density is a local preference, not a
model setting. No optional theme switch is presented without an implemented theme.
See [implementation and verification](FRONTEND_IMPLEMENTATION.md).

The transaction-intelligence scene extends the same system rather than introducing a gaming
or cyberpunk theme. Transaction risk uses size and risk tone; relation types use distinct
geometry and labels. Selection uses a high-contrast ring and contextual dimming. The static
2D equivalent is the default for reduced motion and compact/mobile layouts. The canvas has
no auto-orbit, particles, custom cursor or continuous decorative animation.

## Archived Phase 0 foundation notes

The following records the original scaffold, superseded by the active system above.

## Phase 0 scope

Only the foundation page and a not-found route are implemented. No dashboard, prediction
form, risk visualization, component library or final brand treatment is delivered yet.
The following conventions establish a consistent starting point.

## Tokens

Tailwind theme tokens live in `apps/web/src/styles/index.css` and are used through utility
classes rather than scattering literal colors across components.

| Token       | Value                   | Purpose                                           |
| ----------- | ----------------------- | ------------------------------------------------- |
| `canvas`    | `#f8fafc`               | Application background                            |
| `ink`       | `#0f172a`               | Primary text                                      |
| `muted`     | `#475569`               | Supporting text                                   |
| `brand`     | `#1d4ed8`               | Links, focus and small accents                    |
| `font-sans` | System sans-serif stack | Readable local fonts; no external font dependency |

Use Tailwind spacing on a 4px base, readable line heights, restrained radii and consistent
content widths. The foundation page uses a responsive single column. Final typography,
dark mode, density and risk-state colors remain future design decisions.

## Accessibility and future components

Use semantic landmarks, one primary heading per view, native buttons/links, keyboard
navigation and a visible focus indicator. Pair status color with explicit text/icons.
Target WCAG 2.2 AA contrast and interaction requirements; perform accessibility testing
when interactive screens are introduced. Respect reduced-motion preferences for any future
animation. Avoid communicating uncertainty as a precise validated probability.

Shared primitives belong in `src/components`, domain UI in `src/features`, route views
in `src/pages`, and composition in `src/app`. Future controls must define loading, empty,
error, disabled and focus states. Risk displays must identify score meaning, model version,
missing signals and the explanation target. Do not claim that classifier SHAP explains
the fused score without validation.
