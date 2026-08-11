import { expect, test } from '@playwright/test';

test('registers through the real backend and reads history from the test database', async ({
  page,
}) => {
  const registrationResponse = page.waitForResponse(
    (response) =>
      response.url().includes('/api/auth/register') && response.request().method() === 'POST',
  );

  await page.goto('/register');
  await page.getByLabel('Email').fill('real-e2e@example.com');
  await page.getByLabel('Username').fill('real-e2e-user');
  await page.getByLabel('Password').fill('Secure123');
  await page.getByRole('button', { name: 'Register' }).click();

  expect((await registrationResponse).status()).toBe(201);
  await expect(page).toHaveURL(/\/analyze$/);

  const historyResponse = page.waitForResponse(
    (response) =>
      response.url().includes('/api/v1/history') && response.request().method() === 'GET',
  );
  await page.getByRole('link', { name: 'Dashboard' }).click();

  expect((await historyResponse).status()).toBe(200);
  await expect(page.getByRole('heading', { name: /Welcome back, real-e2e-user/ })).toBeVisible();
  await expect(
    page.getByText('Total Scans').locator('..').getByText('0', { exact: true }),
  ).toBeVisible();
});
