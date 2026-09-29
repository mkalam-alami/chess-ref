import { defineConfig } from 'vitest/config';

// `npm run annot`: fixture annotation tools (tests/tools/annotate.entry.ts; options by environment, see there).
export default defineConfig({
  test: {
    include: ['tests/tools/annotate.entry.ts'],
    disableConsoleIntercept: true,
    reporters: ['dot'],
    testTimeout: 1_800_000,
  },
});
