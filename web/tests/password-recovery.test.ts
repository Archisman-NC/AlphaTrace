import {test} from 'node:test';
import assert from 'node:assert/strict';
import {recoveryCallback,recoveryPassword,recoveryRequestError} from '../src/lib/password-recovery';

test('recovery rejects callback errors without exposing their contents',()=>{
 assert.deepEqual(recoveryCallback('?recovery=1','#error_code=otp_expired&error_description=PRIVATE'),{requested:true,invalid:true});
 assert.deepEqual(recoveryCallback('?error=access_denied',''),{requested:true,invalid:true});
 assert.deepEqual(recoveryCallback('?recovery=invalid',''),{requested:true,invalid:true});
 assert.deepEqual(recoveryCallback('','#type=recovery&access_token=PRIVATE'),{requested:true,invalid:false});
 assert.deepEqual(recoveryCallback('?recovery=1',''),{requested:true,invalid:false});
 assert.deepEqual(recoveryCallback('?view=companies',''),{requested:false,invalid:false});
});
test('recovery requires matching passwords without trimming their meaningful whitespace',()=>{
 assert.throws(()=>recoveryPassword('short','short'),/12 and 128/);
 assert.throws(()=>recoveryPassword('x'.repeat(129),'x'.repeat(129)),/12 and 128/);
 assert.throws(()=>recoveryPassword('a long password','different password'),/do not match/);
 assert.equal(recoveryPassword(' a long password ',' a long password '),' a long password ');
});
test('email request errors distinguish rate limiting and never claim delivery or expose provider text',()=>{
 assert.match(recoveryRequestError({status:429,message:'PRIVATE'}),/Wait/);
 assert.match(recoveryRequestError({status:503,message:'PRIVATE'}),/could not be confirmed/);
 assert.ok(!recoveryRequestError(new Error('PRIVATE')).includes('PRIVATE'));
});
