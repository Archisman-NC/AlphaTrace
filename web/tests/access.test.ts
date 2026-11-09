import { test } from 'node:test';
import assert from 'node:assert/strict';
import { localAccessAllowed, validDevToken, workspaceName } from '../src/lib/access';
test('development bypass fails closed outside explicit loopback development',()=>{
  const env:NodeJS.ProcessEnv={NODE_ENV:'development',ALPHATRACE_LOCAL_DEV:'true'};
  assert.equal(localAccessAllowed(env,'127.0.0.1:3000'),true);
  for(const host of ['evil.example','localhost.evil:3000',null]) assert.equal(localAccessAllowed(env,host),false);
  assert.equal(localAccessAllowed({...env,NODE_ENV:'production'},'localhost:3000'),false);
  assert.equal(localAccessAllowed({...env,ALPHATRACE_LOCAL_DEV:'false'},'localhost:3000'),false);
});
test('development API tokens require a long exact match',()=>{
  const token='x'.repeat(40);
  assert.equal(validDevToken(token,token),true);
  assert.equal(validDevToken('x',token),false);
  assert.equal(validDevToken('short','short'),false);
  assert.equal(validDevToken(token,undefined),false);
});
test('workspace input is trimmed, bounded and rejects controls',()=>{
  assert.equal(workspaceName('  My research  '),'My research');
  for(const value of ['',null,{},' '.repeat(9),'x'.repeat(81),'bad\u0000name']) assert.throws(()=>workspaceName(value));
});
