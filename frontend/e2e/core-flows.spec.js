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
    low_threshold: 0.3,
    high_threshold: 0.6,
    active_model_count: 8,
    failed_model_count: 0,
    version: 'ensemble-e2e',
    fitted: true,
    weight_source: 'e2e-fixture',
  },
  member_outputs: [
    {
      key: 'bert',
      display_name: 'BERT',
      status: 'success',
      raw_score: 0.86,
      calibrated_score: 0.86,
      configured_weight: 1,
      effective_weight: 1,
      weighted_contribution: 0.86,
      error: null,
      error_code: null,
    },
  ],
  xai: {
    status: 'success',
    method: 'occlusion_fallback',
    target: 'ensemble_fake_probability',
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
  await expect(page.getByText('demo-user', { exact: true })).toBeVisible();
});

test('analyses a job advert and records it in the user dashboard', async ({ page }) => {
  await authenticate(page);
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
});

test('shows a recoverable message when the model service is unavailable', async ({ page }) => {
  await authenticate(page);
  await page.route('**/api/v1/analyze', (route) =>
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

  await authenticate(page, '/analyze', adminAuth);
  await page.getByRole('link', { name: /Dashboard/ }).click();

  await expect(page).toHaveURL(/\/dashboard$/);
  await expect(page.getByRole('heading', { name: 'Admin & Research Dashboard' })).toBeVisible();
  await expect(page.getByText('System Healthy')).toBeVisible();
  await expect(page.getByText('Models Deployed')).toBeVisible();
  const requestsBeforeRefresh = healthRequests;
  await page.getByRole('button', { name: 'Refresh system health' }).click();
  await expect.poll(() => healthRequests).toBeGreaterThan(requestsBeforeRefresh);
  await expect(page.getByText('System Healthy')).toBeVisible();
});
