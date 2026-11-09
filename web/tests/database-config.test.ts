import assert from 'node:assert/strict';
import test from 'node:test';
import { databaseConfig } from '../src/lib/database-config.mjs';

test('remote connections require TLS and cannot override certificate verification through URLs', () => {
  const remote = 'postgresql://app:password@db.example.com/database';
  assert.throws(() => databaseConfig(remote, {}), /verified TLS/);
  assert.deepEqual(databaseConfig(remote, { DATABASE_SSL: 'true' }).ssl, { rejectUnauthorized: true });
  for (const param of ['sslmode=disable', 'sslmode=no-verify', 'ssl=false', 'sslrootcert=other']) {
    assert.throws(() => databaseConfig(`${remote}?${param}`, { DATABASE_SSL: 'true' }), /not URL parameters/);
  }
  assert.equal(databaseConfig('postgresql://app@127.0.0.1/local', {}).ssl, false);
  assert.throws(() => databaseConfig(remote, { DATABASE_SSL: 'true', DATABASE_SSL_CA_FILE: '/nonexistent/alphatrace-ca.crt' }));
});
