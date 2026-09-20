import { defineConfig } from "vitest/config";

// Deliberately not vite.config.ts: the app config pulls in the React plugin and the
// dev-server isolation middleware, neither of which a node-side token test needs.
export default defineConfig({
  test: {
    environment: "node",
    include: ["test/**/*.test.ts"],
    // the tokenizer (3.6MB) is fetched from the weights host on first use
    testTimeout: 120_000,
    hookTimeout: 120_000,
  },
});
