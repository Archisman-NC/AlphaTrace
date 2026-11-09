import {test} from 'node:test';import assert from 'node:assert/strict';import {randomUUID} from 'node:crypto';
import {betaCommand,betaEmail,betaToken,newInvitationToken} from '../src/lib/beta-contract';
test('beta validation bounds addresses, capabilities and explicit access revocation',()=>{
 assert.equal(betaEmail(' Person@Example.com '),'person@example.com');for(const email of ['bad','a@b','a b@example.com','a@example.com\nBcc:x@evil.test'])assert.throws(()=>betaEmail(email));
 const token=newInvitationToken();assert.equal(token.length,43);assert.equal(betaToken(token),token);assert.throws(()=>betaToken('short'));
 const id=randomUUID();assert.throws(()=>betaCommand({action:'revoke',id,version:1}));assert.deepEqual(betaCommand({action:'revoke',id,version:1,confirmed:true,owner:'untrusted'}),{action:'revoke',id,version:1});assert.throws(()=>betaCommand({action:'reissue',id,version:0,token}));
});
