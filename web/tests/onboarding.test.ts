import { test } from 'node:test';
import assert from 'node:assert/strict';
import { validateOnboarding,rankFirstReviews,type FirstReview } from '../src/lib/onboarding-contract';
test('setup rejects unconfirmed imports, malformed quantities, unknown topics and stale version shapes',()=>{
 const base={action:'companies',version:0,portfolioVersion:0,symbols:['TCS'],holdings:[{symbol:'INFY',quantity:'2.123456'}]};
 assert.throws(()=>validateOnboarding(base),/Confirm/);
 assert.deepEqual(validateOnboarding({...base,importConfirmed:true}),{...base,importConfirmed:true});
 assert.throws(()=>validateOnboarding({...base,importConfirmed:true,holdings:[{symbol:'INFY',quantity:'2e4'}]}),/positive quantity/);
 assert.throws(()=>validateOnboarding({action:'topics',version:0,topics:['buy']}),/supported/);
 assert.throws(()=>validateOnboarding({action:'skip',version:-1}),/Reload/);
 assert.throws(()=>validateOnboarding({...base,symbols:['UNKNOWN']}),/universe/);
 assert.deepEqual(validateOnboarding({action:'topics',version:1,topics:[]}),{action:'topics',version:1,topics:[]});
});
test('first comparison prioritizes explicit interests then recency without filtering other companies',()=>{
 const reviews:FirstReview[]=[{id:'1',symbol:'TCS',title:'One',periodEnd:'2026-06-30',publishedAt:'2026-11-08',metrics:['revenue']},{id:'2',symbol:'INFY',title:'Two',periodEnd:'2026-06-30',publishedAt:'2026-11-01',metrics:['debt']}];
 assert.deepEqual(rankFirstReviews(reviews,['debt']).map(r=>r.id),['2','1']);
 assert.deepEqual(rankFirstReviews(reviews,[]).map(r=>r.id),['1','2']);
 assert.equal(reviews[0].id,'1');
});
