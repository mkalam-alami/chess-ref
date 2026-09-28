import { defineConfig } from 'vitest/config';

export default defineConfig({
  base: '/chess-ref/',
  worker: { format: 'es' },
  build: { target: 'es2022', chunkSizeWarningLimit: 12000 },
  test: { include: ['tests/**/*.test.ts'] },
});
