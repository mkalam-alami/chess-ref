import { defineConfig } from 'vitest/config';

// `npm run bench`: synthetic + real-photo detection benchmark (not part of `npm test`).
export default defineConfig({
  test: {
    include: ['tests/synth/bench.entry.ts'],
    disableConsoleIntercept: true,
    reporters: ['dot'],
    testTimeout: 900_000,
  },
});
