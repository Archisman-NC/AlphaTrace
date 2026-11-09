import {test} from 'node:test';
import assert from 'node:assert/strict';
import {deletionInput,passwordInput} from '../src/lib/account-contract';
test('deletion requires exact confirmation, acknowledgement and a bounded current password',()=>{
 const input={confirmation:'DELETE MY ACCOUNT',acknowledged:true,password:'synthetic'};
 assert.deepEqual(deletionInput({...input,owner:'someone else'}),{password:'synthetic'});
 for(const change of [{confirmation:'delete my account'},{acknowledged:false},{password:''},{password:'x'.repeat(129)}])assert.throws(()=>deletionInput({...input,...change}));
 assert.throws(()=>deletionInput(null));
});
test('password changes require current password, matching new values and length limits',()=>{
 const input={currentPassword:'old password',password:'new strong password',confirmPassword:'new strong password'};assert.equal(passwordInput(input).password,input.password);
 for(const change of [{currentPassword:''},{password:'short'},{confirmPassword:'different'},{password:'x'.repeat(129)},{currentPassword:input.password}])assert.throws(()=>passwordInput({...input,...change}));
});
