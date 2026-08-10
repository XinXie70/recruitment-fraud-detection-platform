import { expect, test } from '@playwright/test';

const AUTH_RESPONSE = {
  access_token: 'e2e-token',
  token_type: 'bearer',
  user: { id: 7, email: 'demo@example.com', username: 'demo-user', is_admin: false },
};

const ANALYSIS_RESPONSE = {
  api_version: '1.0',
  status: 'success',
  job_relevance_score: 0.98,
  ensemble: {
    status: 'success',
    risk_score: 0.86,
    classification_label: 'Likely Deceptive',
    risk_level: 'high',
    prediction: 'fake',
    recommended_action: 'High Risk Warning',
    low_threshold: 0.15,
    high_threshold: 0.32,
    active_model_count: 2,
    failed_model_count: 0,
    version: 'bert-lr-fp-gate-e2e',
    fitted: true,
    decision_strategy: 'bert_primary_lr_fp_gate',
    method: 'bert_lr_fp_gate',
    bert_low_threshold: 0.15,
    bert_high_threshold: 0.32,
    lr_gate_threshold: 0.06,
    risk_score_source: 'bert',
    gate_triggered: false,
  },
  member_outputs: [
    {
      key: 'bert',
      display_name: 'BERT',
      status: 'success',
      raw_score: 0.86,
      role: 'primary_score',
      decision_active: true,
      error: null,
      error_code: null,
    },
    {
      key: 'lr',
      display_name: 'Logistic Regression',
      status: 'success',
      raw_score: 0.72,
      role: 'false_positive_gate',
      decision_active: false,
      error: null,
      error_code: null,
    },
  ],
  xai: {
    status: 'success',
    method: 'shap_partition',
    target: 'ensemble_risk_score',
    version: 'xai-e2e',
    base_value: 0.2,
    output_value: 0.86,
    items: [],
    message: null,
  },
  gentle_ai: {
    status: 'success',
    provider: 'template',
    summary: 'This advertisement contains several high-risk signals.',
    evidence_explanations: [],
    next_steps: ['Do not send money or identity documents.'],
    learning_item_ids: [],
    disclaimer: 'This result supports, but does not replace, human judgement.',
    message: null,
    version: 'gentle-e2e',
  },
  url_analysis: {
    urls_found: 0,
    risk_score: 0,
    risk_level: 'low',
    high_risk_count: 0,
    medium_risk_count: 0,
    urls: [],
    reasons: [],
  },
};

async function expectResponsiveViewport(page) {
  await expect
    .poll(() =>
      page.evaluate(
        () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
      ),
    )
    .toBeLessThanOrEqual(0);

  await expect(page.locator('nav.app-nav')).toBeVisible();
}

async function authenticate(page, expectedPath = '/analyze', authResponse = AUTH_RESPONSE) {
  await page.route('**/api/auth/login', async (route) => {
    const request = route.request();
    expect(request.method()).toBe('POST');
    expect(request.postDataJSON()).toEqual({
      identifier: 'demo-user',
      password: 'Secure123',
    });
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(authResponse),
    });
  });

  await page.goto('/login');
  await page.getByLabel('Email or username').fill('demo-user');
  await page.getByLabel('Password').fill('Secure123');
  await page.getByRole('button', { name: 'Log in' }).click();
  await expect(page).toHaveURL(new RegExp(`${expectedPath}$`));
}

test.beforeEach(async ({ page }) => {
  await page.route('**/api/v1/history?*', async (route) => {
    expect(route.request().headers().authorization).toBe('Bearer e2e-token');
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        items: [],
        total: 0,
        page: 1,
        page_size: 100,
        total_pages: 0,
      }),
    });
  });
  await page.goto('/');
  await page.evaluate(() => {
    window.localStorage.clear();
    window.sessionStorage.clear();
  });
});

test('redirects a signed-out user to login and returns to the requested page', async ({ page }) => {
  await page.goto('/dashboard');
  await expect(page).toHaveURL(/\/login$/);
  await authenticate(page, '/dashboard');
  await expect(page).toHaveURL(/\/dashboard$/);
  await expect(page.getByRole('heading', { name: /Welcome back, demo-user/ })).toBeVisible();
  await expectResponsiveViewport(page);
});

test('registers a new user and opens the analyser', async ({ page }) => {
  await page.route('**/api/auth/register', async (route) => {
    expect(route.request().postDataJSON()).toEqual({
      email: 'demo@example.com',
      username: 'demo-user',
      password: 'Secure123',
    });
    await route.fulfill({
      status: 201,
      contentType: 'application/json',
      body: JSON.stringify(AUTH_RESPONSE),
    });
  });

  await page.goto('/register');
  await page.getByLabel('Email').fill('demo@example.com');
  await page.getByLabel('Username').fill('demo-user');
  await page.getByLabel('Password').fill('Secure123');
  await page.getByRole('button', { name: 'Register' }).click();

  await expect(page).toHaveURL(/\/analyze$/);
  await expect(page.getByRole('heading', { name: 'Detect Fake Job Advertisements' })).toBeVisible();
  await expectResponsiveViewport(page);
});

test('analyses a job advert and records it in the user dashboard', async ({ page }) => {
  await authenticate(page);
  await page.route('**/api/v1/analyze/score', async (route) => {
    expect(route.request().headers().authorization).toBe('Bearer e2e-token');
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ...ANALYSIS_RESPONSE,
        phase: 'score',
        xai: {
          status: 'unavailable',
          method: 'unavailable',
          items: [],
          message: 'Detailed model-derived explanation is being prepared.',
        },
      }),
    });
  });
  await page.route('**/api/v1/analyze', async (route) => {
    expect(route.request().headers().authorization).toBe('Bearer e2e-token');
    expect(route.request().postDataJSON().text).toContain('bank details');
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(ANALYSIS_RESPONSE),
    });
  });

  await page
    .getByLabel('Job Advertisement Text')
    .fill(
      'We offer an immediate remote position with guaranteed income. Send your bank details and identity documents before a formal interview.',
    );
  await page.getByRole('button', { name: 'Analyze Text' }).click();

  await expect(page.getByRole('heading', { name: 'High Risk Warning' })).toBeVisible();
  await expect(page.getByText('86', { exact: true })).toBeVisible();
  await page.getByRole('link', { name: 'Dashboard' }).click();
  await expect(
    page.getByText('Total Scans').locator('..').getByText('1', { exact: true }),
  ).toBeVisible();
  await expect(page.getByRole('cell', { name: 'Likely Deceptive' })).toBeVisible();
  await expectResponsiveViewport(page);
});

test('shows a recoverable message when the model service is unavailable', async ({ page }) => {
  await authenticate(page);
  await page.route('**/api/v1/analyze/score', (route) =>
    route.fulfill({
      status: 503,
      contentType: 'application/json',
      body: JSON.stringify({ detail: 'Prediction service is temporarily unavailable.' }),
    }),
  );

  await page
    .getByLabel('Job Advertisement Text')
    .fill(
      'This is a complete software engineering job advertisement with salary, responsibilities, requirements, company details, and a formal interview process.',
    );
  await page.getByRole('button', { name: 'Analyze Text' }).click();

  await expect(page.getByText('Analysis Failed')).toBeVisible();
  await expect(
    page.getByText('The analysis service is temporarily unavailable. Please try again shortly.'),
  ).toBeVisible();
  await expect(page.getByRole('button', { name: 'Analyze Text' })).toBeEnabled();
  await expectResponsiveViewport(page);
});

test('routes an administrator to the research dashboard and reports service health', async ({
  page,
}) => {
  const adminAuth = {
    ...AUTH_RESPONSE,
    user: { ...AUTH_RESPONSE.user, username: 'admin-user', is_admin: true },
  };
  let healthRequests = 0;
  await page.route('**/api/health', (route) => {
    healthRequests += 1;
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ status: 'healthy' }),
    });
  });
  await page.route('**/api/admin/stats', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        total_users: 12,
        total_analyses: 48,
        analyses_today: 7,
        avg_risk_score: 0.25,
        high_risk_count: 8,
        medium_risk_count: 10,
        low_risk_count: 30,
      }),
    }),
  );
  await page.route('**/api/admin/model-metrics', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        version: '2026.07',
        dataset: 'EMS CAD held-out test set',
        models: [
          {
            model: 'BERT',
            accuracy: 0.99,
            precision: 0.92,
            recall: 0.89,
            f1: 0.905,
            threshold: 0.5,
            category: 'transformer',
          },
        ],
      }),
    }),
  );

  await authenticate(page, '/analyze', adminAuth);
  await page.getByRole('link', { name: /Dashboard/ }).click();

  await expect(page).toHaveURL(/\/dashboard$/);
  await expect(page.getByRole('heading', { name: 'Admin Analytics Dashboard' })).toBeVisible();
  await expect(page.getByText('System Healthy')).toBeVisible();
  await expect(page.getByText('Total Users').locator('..').getByText('12')).toBeVisible();
  await expect(page.getByText('Models Deployed')).toBeVisible();
  const requestsBeforeRefresh = healthRequests;
  await page.getByRole('button', { name: 'Refresh system health' }).click();
  await expect.poll(() => healthRequests).toBeGreaterThan(requestsBeforeRefresh);
  await expect(page.getByText('System Healthy')).toBeVisible();
  await expectResponsiveViewport(page);
});
