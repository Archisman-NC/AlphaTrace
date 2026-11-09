import test from 'node:test';
import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { matchAssumptions,validateReview,revisitTimestamp,inboxCursor,inboxFilter } from '../src/lib/inbox-contract';
import { fixture } from './research-fixture';
import type { Assumption } from '../src/lib/thesis-input';
const assumption=(metric:string):Assumption=>({id:randomUUID(),statement:'Revenue will grow; operating margin will hold.',metric,baseline:'Unknown',direction:'up',horizon:'Two quarters',reviewCondition:'My own unparsed threshold'});
test('matching uses explicit metric topics, word boundaries and no inferred belief or threshold',()=>{
 const a=assumption('Revenue growth'),b=assumption('Operating profit'),c=assumption(''),d=assumption('presales productivity');
 const result=matchAssumptions([a,b,c,d],fixture());
 assert.deepEqual(result.map(r=>r.metrics),[['revenue'],['operating_profit']]);
 assert.deepEqual(result[0].assumption,a);
 assert.deepEqual(matchAssumptions([assumption('margin')],fixture()),[],'ambiguous topics must not be guessed');
 const bank=fixture();bank.comparisons[0].metric='net_interest_income';bank.comparisons[1].metric='net_interest_margin';
 assert.deepEqual(matchAssumptions([assumption('NII and NIM')],bank)[0].metrics,['net_interest_income','net_interest_margin']);
});
test('review commands validate decisions, IDs, bounds and explicitly saved revisions',()=>{
 const c={updateId:randomUUID(),commandId:randomUUID(),version:0,action:'keep',note:'Keep my view',owner_id:'not trusted'};
 assert.equal('owner_id' in validateReview(c),false);
 for(const patch of [{version:-1},{version:2147483647},{note:'x'},{note:'x'.repeat(2001)},{action:'buy'},{updateId:'bad'},{action:'revise'},{action:'defer',deferUntil:'2026-02-30'}])assert.throws(()=>validateReview({...c,...patch}));
 assert.equal(validateReview({...c,action:'revise',thesisVersion:2}).thesisVersion,2);
 assert.equal(validateReview({...c,thesisVersion:2}).thesisVersion,null);
});
test('deferred reviews use 9 am IST, future dates and a bounded horizon',()=>{
 const now=Date.parse('2026-11-03T00:00:00Z');
 assert.equal(revisitTimestamp('2026-11-04',now),'2026-11-04T03:30:00.000Z');
 assert.throws(()=>revisitTimestamp('2026-11-02',now));
 assert.throws(()=>revisitTimestamp('2028-11-03',now));
});
test('pagination retains PostgreSQL microseconds so tied batch items are not skipped',()=>{
 const id=randomUUID(),stamp='2026-11-03T03:04:05.123456Z';
 assert.deepEqual(inboxCursor(stamp+'|'+id),[stamp,id]);
 assert.throws(()=>inboxCursor('invalid|'+id));assert.throws(()=>inboxFilter('unknown'));
 assert.equal(inboxFilter(null),'pending');
});
