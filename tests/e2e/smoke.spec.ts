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

  // With the panel closed no debug image may be drawn over the live video (it would lag and ghost).
  await page.waitForTimeout(1500);
  await expect(page.locator('#overlay')).toHaveAttribute('data-debug-image', 'off');

  // No board profile is locked without a board in view, so the reset button stays hidden.
  await expect(page.locator('#resetBoard')).toBeHidden();

  // Every result carries an occupancy field: null (no board / dropped) or a 64-cell grid.
  await expect(app).toHaveAttribute('data-occupancy', /^(null|u8:64)$/, { timeout: 30_000 });

  // Frames flow through the worker: open the panel and wait for stage timings.
  await page.click('.dbg-toggle');
  await expect(page.locator('.dbg-panel pre')).toContainText('canny', { timeout: 30_000 });
  await page.selectOption('.dbg-panel select >> nth=0', 'edges');
  await expect(page.locator('#overlay')).toHaveAttribute('data-debug-image', 'on', { timeout: 15_000 });
  await page.waitForTimeout(500);
  await page.screenshot({ path: 'test-results/smoke.png' });

  // Closing the panel clears the debug image again.
  await page.click('.dbg-toggle');
  await expect(page.locator('#overlay')).toHaveAttribute('data-debug-image', 'off', { timeout: 15_000 });

  expect(errors).toEqual([]);
});
