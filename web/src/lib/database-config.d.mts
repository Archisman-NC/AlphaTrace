import type { ClientConfig } from 'pg';
export function databaseConfig(connectionString: string | undefined, env?: Record<string, string | undefined>): ClientConfig;
