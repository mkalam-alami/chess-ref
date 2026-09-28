import { expect, test } from '@playwright/test';

test('camera starts, worker loads OpenCV, no console errors', async ({ page }) => {
  const errors: string[] = [];
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  page.on('pageerror', (e) => errors.push(String(e)));

  await page.goto('./');
  await page.click('#startBtn');

  const app = page.locator('#app');
  await expect(app).toHaveAttribute('data-status', 'running');
  await expect(app).toHaveAttribute('data-worker', 'ready', { timeout: 60_000 });
  await expect(page.locator('video.media')).toBeVisible();

  // Frames flow through the worker: open the panel and wait for stage timings.
  await page.click('.dbg-toggle');
  await expect(page.locator('.dbg-panel pre')).toContainText('canny', { timeout: 30_000 });
  await page.selectOption('.dbg-panel select >> nth=0', 'edges');
  await page.waitForTimeout(1000);
  await page.screenshot({ path: 'test-results/smoke.png' });

  expect(errors).toEqual([]);
});
