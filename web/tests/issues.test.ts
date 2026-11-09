import {test} from 'node:test';
import assert from 'node:assert/strict';
import {randomUUID} from 'node:crypto';
import {validateIssue,issueQuery} from '../src/lib/issue-contract';
const submit={action:'submit',commandId:randomUUID(),publicationId:randomUUID(),metric:'revenue',category:'incorrect_value',note:'  Wrong figure in this row.  '};
test('reports validate bounded text and strip client ownership/context',()=>{
 const result=validateIssue({...submit,owner_id:randomUUID(),context:{symbol:'FAKE'},status:'resolved'});
 assert.equal(result.note,'Wrong figure in this row.');assert.equal('context' in result,false);assert.equal('owner_id' in result,false);
 for(const change of [{note:'hi'},{note:'x'.repeat(2001)},{note:'bad\0text'},{category:'__proto__'},{metric:'__proto__'},{publicationId:'bad'},{commandId:'bad'}])assert.throws(()=>validateIssue({...submit,...change}));
 assert.equal(validateIssue({...submit,metric:null}).action,'submit');
});
test('resolution requires a valid outcome, response, version and correction ID',()=>{
 const command={action:'resolve',id:randomUUID(),commandId:randomUUID(),version:1,note:'Explained against the source.',outcome:'explained',correctionId:null};
 assert.equal(validateIssue(command).action,'resolve');
 for(const change of [{note:'short'},{version:-1},{version:1.2},{outcome:'unknown'},{outcome:'corrected',correctionId:null}])assert.throws(()=>validateIssue({...command,...change}));
 assert.equal(validateIssue({...command,action:'in_review',note:''}).note,'Review started.');
 const reopened=validateIssue({...command,action:'reopen'});assert.ok(reopened.action!=='submit');assert.equal(reopened.outcome,null);
});
test('queue filters and precision-preserving cursors are bounded',()=>{
 const cursor='2026-11-09T10:00:00.123456Z|'+randomUUID();assert.equal(issueQuery({cursor}).cursor?.[0],'2026-11-09T10:00:00.123456Z');
 assert.deepEqual(issueQuery({}),{scope:'mine',status:'active',cursor:null,report:null});
 for(const q of [{scope:'everyone'},{status:'draft'},{report:'bad'},{cursor:'bad'},{cursor:'2026-11-09Tbad|'+randomUUID()}])assert.throws(()=>issueQuery(q));
});
