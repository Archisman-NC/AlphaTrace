import { logger, task } from "@trigger.dev/sdk";

/** Registration/execution check; no provider requests or database mutations. */
export const alphaTraceHealthCheck = task({
  id: "alphatrace-health-check",
  maxDuration: 30,
  retry: { maxAttempts: 1 },
  run: async (_payload: Record<string, never>) => {
    logger.info("AlphaTrace background task executed successfully.");
    return {
      application: "AlphaTrace",
      status: "ok",
      checkedAt: new Date().toISOString(),
    };
  },
});
