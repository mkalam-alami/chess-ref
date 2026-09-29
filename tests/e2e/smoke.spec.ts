import { expect, test } from '@playwright/test';

test('camera starts, worker loads OpenCV, no console errors', async ({ page }) => {
  const errors: string[] = [];
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  page.on('pageerror', (e) => errors.push(String(e)));

  await page.goto('./');
  // The move sidebar is hidden on the start screen.
  await expect(page.locator('#moves')).toBeHidden();
  await page.click('#startBtn');

  const app = page.locator('#app');
  await expect(app).toHaveAttribute('data-status', 'running');
  await expect(app).toHaveAttribute('data-worker', 'ready', { timeout: 60_000 });
  await expect(page.locator('video.media')).toBeVisible();

  // With the panel closed no debug image may be drawn over the live video (it would lag and ghost).
  await page.waitForTimeout(1500);
  await expect(page.locator('#overlay')).toHaveAttribute('data-debug-image', 'off');

  // The fake camera is a single device: no camera picker on the start screen or in the debug panel.
  await expect(app).toHaveAttribute('data-cameras', '1');
  await expect(page.locator('#cameraPick')).toBeHidden();
  await expect(page.locator('.dbg-camera')).toBeHidden();

  // Every result carries an occupancy field: null (no board / dropped) or a 64-cell grid.
  await expect(app).toHaveAttribute('data-occupancy', /^(null|u8:64)$/, { timeout: 30_000 });

  // Game layer: the sidebar is shown with a status line, and no game can have started without a board in view
  // ('playing' is allowed in case a saved game was restored).
  await expect(app).toHaveAttribute('data-game', /^(waiting|playing)$/);
  const moves = page.locator('#moves');
  await expect(moves).toBeVisible();
  await expect(moves.locator('.mv-status')).not.toBeEmpty();
  await expect(moves.locator('.mv-copy')).toBeVisible();
  // The sound toggle mutes and unmutes (persisted); the audio context was created on the Start tap without errors.
  const soundBtn = moves.locator('.mv-sound');
  await expect(soundBtn).toHaveAttribute('aria-pressed', 'true');
  await soundBtn.click();
  await expect(soundBtn).toHaveAttribute('aria-pressed', 'false');
  expect(await page.evaluate(() => localStorage.getItem('chess-ref.muted'))).toBe('1');
  await soundBtn.click();
  await expect(soundBtn).toHaveAttribute('aria-pressed', 'true');
  // Collapsing keeps only the toggle; expanding restores the list.
  await moves.locator('.mv-toggle').click();
  await expect(moves.locator('.mv-copy')).toBeHidden();
  await moves.locator('.mv-toggle').click();
  await expect(moves.locator('.mv-copy')).toBeVisible();

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
