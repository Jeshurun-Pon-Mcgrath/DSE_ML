import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { chromium } from 'playwright';
import {
  explanation,
  final,
  health,
  model,
  prediction,
  scenario,
  transaction,
} from './fixtures.mjs';
const server = spawn(
  process.execPath,
  ['../../node_modules/vite/bin/vite.js', '--host', '127.0.0.1', '--port', '5175', '--strictPort'],
  { stdio: 'pipe' },
);
let browser;
let passed = 0;
async function check(name, fn) {
  await fn();
  passed++;
  console.log('PASS', name);
}
const origin = 'http://127.0.0.1:5175';
try {
  for (let i = 0; i < 80; i++) {
    try {
      await fetch(origin);
      break;
    } catch {
      await new Promise((r) => setTimeout(r, 250));
    }
  }
  const candidates = [
    process.env.RISKBYTE_BROWSER_PATH,
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  ];
  const path = candidates.find((p) => p && existsSync(p));
  browser = await chromium.launch({
    ...(path ? { executablePath: path } : {}),
    headless: true,
    args: [
      '--no-sandbox',
      '--enable-webgl',
      '--use-angle=swiftshader',
      '--enable-unsafe-swiftshader',
    ],
  });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  await page.addInitScript(() => {
    Object.defineProperty(navigator, 'hardwareConcurrency', { configurable: true, value: 8 });
    Object.defineProperty(navigator, 'deviceMemory', { configurable: true, value: 8 });
  });
  let mode = 'normal',
    state = 'UNREVIEWED';
  let pageError;
  let submitted;
  page.on('pageerror', (e) => {
    pageError = e;
  });
  await page.route('**/api/v1/**', async (route) => {
    const url = new URL(route.request().url()),
      p = url.pathname.replace('/api/v1', '');
    if (mode === 'loading' && p === '/dashboard/summary')
      await new Promise((r) => setTimeout(r, 1500));
    if (mode === 'error' && p === '/dashboard/summary') {
      await route.fulfill({
        status: 503,
        json: { error: { message: 'Synthetic fixture API unavailable' } },
      });
      return;
    }
    let data;
    if (p === '/system/health') data = health;
    else if (p === '/system/model') data = model;
    else if (p === '/evaluation/final') data = final;
    else if (p === '/evaluation/scenarios')
      data = { VELOCITY_BURST: scenario, LOW_AND_SLOW: scenario };
    else if (p === '/evaluation/hard-negatives')
      data = {
        SHOPPING_FESTIVAL: {
          count: 20,
          supervised: { count: 20, flagged: 0, rate: 0 },
          abrf: { count: 20, flagged: 1, rate: 0.05 },
        },
      };
    else if (p === '/dashboard/summary')
      data = {
        transaction_count: 210,
        investigation_count: 8,
        average_risk_score: 0.1,
        risk_level_distribution: { LOW: 202, ELEVATED: 8 },
        risk_trend: [{ elapsed_day_index: 1, transactions: 210, average_risk_score: 0.1 }],
        recent_transaction_activity: [
          transaction,
          {
            ...transaction,
            transaction: {
              ...transaction.transaction,
              TransactionID: 'ui-fixture-2',
              behavioural_entity_id: 'ui-entity-2',
              merchant_id: 'ui-merchant-2',
              device_id: null,
            },
            prediction: {
              ...transaction.prediction,
              transaction_id: 'ui-fixture-2',
              risk: {
                score: 0.08,
                level: 'LOW',
                investigation_required: false,
              },
            },
          },
        ],
        recent_high_risk_transactions: [transaction],
        scope: 'Synthetic UI fixture',
      };
    else if (p === '/transactions') {
      data = {
        total: 30,
        page: Number(url.searchParams.get('page') ?? 1),
        page_size: 15,
        items: [{ ...transaction, investigation: state }],
      };
    } else if (p.endsWith('/explanation')) data = explanation;
    else if (p.endsWith('/investigation')) {
      state = route.request().postDataJSON().state;
      data = { transaction_id: 'ui-fixture-1', state, scores_unchanged: true };
    } else if (p.startsWith('/transactions/')) data = { ...transaction, investigation: state };
    else if (p === '/audit')
      data = {
        total: 1,
        page: 1,
        page_size: 25,
        items: [
          {
            id: 1,
            transaction_id: 'ui-fixture-1',
            action: 'prediction_generated',
            timestamp: '2026-01-01T00:00:00Z',
            metadata: {},
          },
        ],
      };
    else if (p === '/abrf/config')
      data = {
        status: 'abrf_ready',
        data: {
          configuration: {
            version: 'abrf-v1',
            weights: { wp: 0.9, wa: 0, wb: 0.1 },
            policy: 'BALANCED',
            decision_threshold: 0.29755749437222373,
            risk_boundaries: [0.01, 0.1, 0.5, 0.9],
          },
        },
      };
    else if (p === '/predict') {
      submitted = route.request().postDataJSON();
      data = { ...prediction, transaction_id: submitted.TransactionID };
    } else data = { status: 'dataset_not_configured', data: null };
    await route.fulfill({ json: data });
  });
  await check('dashboard API data and final metric source', async () => {
    await page.goto(origin + '/app');
    await page.getByText('210', { exact: true }).first().waitFor();
    await page.getByText('80%', { exact: true }).waitFor();
    assert.equal(await page.getByRole('heading', { name: 'Risk overview' }).count(), 1);
  });
  await check(
    '3D relationship network, transaction selection and analytical fallback',
    async () => {
      await page.getByRole('heading', { name: '3D transaction intelligence' }).waitFor();
      const open3d = page.getByRole('button', { name: 'Open 3D' });
      if (await open3d.isVisible().catch(() => false)) await open3d.click();
      await page.locator('.network-stage').scrollIntoViewIfNeeded();
      await page.locator('.network-stage canvas').waitFor();
      await page.getByRole('button', { name: /ui-fixture-2/ }).click();
      assert.equal(await page.locator('.network-inspector h3').innerText(), 'ui-fixture-2');
      assert.match(
        await page.locator('.network-inspector').innerText(),
        /ABRF anomaly contribution\s*0/,
      );
      await page.getByRole('button', { name: 'Use 2D' }).click();
      await page.getByTestId('transaction-network-fallback').waitFor();
      assert.match(
        await page.locator('.network-caption').innerText(),
        /2 transaction nodes and 5 observed relationship edges/,
      );
    },
  );
  await check('reduced-motion preference starts with the useful 2D network', async () => {
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.goto(origin + '/app');
    await page.getByTestId('transaction-network-fallback').waitFor();
    assert.match(
      await page.locator('.network-fallback-note').innerText(),
      /Reduced motion is enabled/,
    );
    await page.emulateMedia({ reducedMotion: 'no-preference' });
  });
  await check('transaction table and risk filters', async () => {
    await page.goto(origin + '/app/transactions');
    await page.getByRole('link', { name: 'ui-fixture-1' }).first().waitFor();
    const filteredRequest = page.waitForRequest(
      (r) => new URL(r.url()).searchParams.get('risk_level') === 'HIGH',
    );
    await page.getByLabel('Risk level', { exact: true }).selectOption('HIGH');
    assert.match((await filteredRequest).url(), /risk_level=HIGH/);
  });
  await check('search, sorting and pagination', async () => {
    const sortedRequest = page.waitForRequest((r) => {
      const q = new URL(r.url()).searchParams;
      return q.get('search') === 'ui-fixture' && q.get('sort') === 'risk';
    });
    await page.getByLabel('Search transaction ID').fill('ui-fixture');
    await page.getByLabel('Sort by').selectOption('risk');
    await sortedRequest;
    const nextRequest = page.waitForRequest(
      (r) => new URL(r.url()).searchParams.get('page') === '2',
    );
    await page.getByRole('button', { name: 'Next page' }).click();
    const query = (await nextRequest).url();
    assert.match(query, /page=2/);
    assert.match(query, /sort=risk/);
    assert.match(query, /search=ui-fixture/);
  });
  await check('transaction detail and investigation mutation', async () => {
    await page.goto(origin + '/app/transactions/ui-fixture-1');
    await page.getByLabel('Review state').selectOption('UNDER_REVIEW');
    await page.getByRole('button', { name: 'Save review' }).click();
    await page.getByText('Review saved. Model scores remain unchanged.').waitFor();
    assert.equal(state, 'UNDER_REVIEW');
  });
  await check(
    'four explanation sections, genuine contract scope and zero anomaly messaging',
    async () => {
      await page.goto(origin + '/app/transactions/ui-fixture-1/explanation');
      await page.getByRole('heading', { name: 'Exact ABRF calculation' }).waitFor();
      for (const name of [
        'Risk-increasing classifier factors',
        'Strictly prior behavioural evidence',
        'Anomaly diagnostic',
      ])
        assert.equal(await page.getByRole('heading', { name }).count(), 1);
      assert.match(await page.locator('main').innerText(), /ABRF contribution = 0/);
      assert.match(await page.locator('main').innerText(), /underlying Decision Tree/);
    },
  );
  await check('ABRF configuration and threshold', async () => {
    await page.goto(origin + '/app/abrf');
    await page.getByRole('heading', { name: 'Frozen active configuration' }).waitFor();
    assert.match(await page.locator('main').innerText(), /0.29755749437222373/);
  });
  await check('final scenario performance', async () => {
    await page.goto(origin + '/app/performance/scenarios');
    await page.getByRole('table', { name: 'Final scenario performance' }).waitFor();
    assert.match(await page.locator('main').innerText(), /Velocity Burst/);
  });
  await check('hard-negative performance', async () => {
    await page.goto(origin + '/app/performance/hard-negatives');
    await page.getByRole('table', { name: 'Hard-negative results' }).waitFor();
  });
  await check('audit, system and settings routes', async () => {
    for (const path of ['/app/audit', '/app/system', '/app/settings']) {
      await page.goto(origin + path);
      await page.locator('h1').waitFor();
      assert.equal(await page.getByRole('alert').count(), 0);
    }
  });
  await check('public, legal, technology and 404 routes', async () => {
    for (const path of ['/', '/technology', '/privacy', '/terms', '/missing-route']) {
      await page.goto(origin + path);
      await page.locator('h1').waitFor();
    }
    assert.match(await page.locator('h1').innerText(), /Page not found/);
  });
  await check('single prediction form validates and submits outcome-free context', async () => {
    await page.goto(origin + '/app/transactions');
    await page.getByRole('button', { name: 'New prediction' }).click();
    await page.getByLabel('Transaction ID', { exact: true }).fill('ui-submission');
    await page.getByLabel('Behavioural entity ID', { exact: true }).fill('ui-entity');
    await page.getByLabel('Amount (INR)', { exact: true }).fill('500');
    await page.getByLabel('Elapsed seconds', { exact: true }).fill('100001');
    await page.getByRole('button', { name: 'Generate prediction' }).click();
    await page.waitForURL('**/app/transactions/ui-submission');
    assert.equal(submitted.TransactionID, 'ui-submission');
    assert.equal(submitted.TransactionAmt, 500);
    assert.equal('isFraud' in submitted, false);
    assert.equal(submitted.is_international, null);
  });
  await check('batch form enforces the frontend size boundary', async () => {
    await page.goto(origin + '/app/transactions');
    await page.getByRole('button', { name: 'New prediction' }).click();
    await page.getByRole('button', { name: 'JSON batch', exact: true }).click();
    await page
      .getByLabel('Transaction objects (JSON array)')
      .fill(JSON.stringify(Array.from({ length: 501 }, () => transaction.transaction)));
    await page.getByRole('button', { name: 'Submit bounded batch' }).click();
    await page.getByRole('alert').waitFor();
    assert.match(await page.getByRole('alert').innerText(), /1–500/);
    await page.getByRole('button', { name: 'Close dialog' }).click();
  });
  await check('loading state', async () => {
    mode = 'loading';
    await page.goto(origin + '/app');
    await page.getByRole('status', { name: 'Loading data' }).first().waitFor();
  });
  await check('error state', async () => {
    mode = 'error';
    await page.goto(origin + '/app');
    await page.getByRole('alert').first().waitFor();
    assert.match(await page.locator('main').innerText(), /API unavailable/);
    mode = 'normal';
  });
  await check('mobile navigation keyboard close and overflow', async () => {
    await page.setViewportSize({ width: 430, height: 900 });
    await page.goto(origin + '/app/transactions');
    await page.getByRole('button', { name: 'Open navigation' }).click();
    await page.getByRole('dialog', { name: 'Workspace navigation' }).waitFor();
    await page.keyboard.press('Escape');
    assert.equal(await page.getByRole('dialog').count(), 0);
    assert.equal(
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
      true,
    );
  });
  assert.equal(pageError, undefined);
  console.log(`${passed} browser integration tests passed; synthetic UI fixtures only.`);
} finally {
  await browser?.close();
  server.kill();
}
