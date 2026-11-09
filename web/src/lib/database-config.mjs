import { readFileSync } from 'node:fs';

// Shared by the server, migration CLI and integration checks. URL parameters
// must not silently replace certificate verification in node-postgres.
export function databaseConfig(connectionString, env = process.env) {
  if (!connectionString) throw new Error('Database is not configured.');
  const url = new URL(connectionString);
  if (![ 'postgres:', 'postgresql:' ].includes(url.protocol)) throw new Error('Invalid database protocol.');
  if ([...url.searchParams.keys()].some(key => key.toLowerCase().startsWith('ssl'))) {
    throw new Error('Configure database TLS through DATABASE_SSL and DATABASE_SSL_CA_FILE, not URL parameters.');
  }
  const local = ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname);
  if (!local && env.DATABASE_SSL !== 'true') throw new Error('Remote database connections require verified TLS.');
  return {
    connectionString,
    connectionTimeoutMillis: 10000,
    ssl: env.DATABASE_SSL === 'true' ? {
      rejectUnauthorized: true,
      ...(env.DATABASE_SSL_CA_PEM ? { ca: env.DATABASE_SSL_CA_PEM } : env.DATABASE_SSL_CA_FILE ? { ca: readFileSync(env.DATABASE_SSL_CA_FILE, 'utf8') } : {}),
    } : false,
  };
}
