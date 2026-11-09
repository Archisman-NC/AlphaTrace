import test from 'node:test';
import assert from 'node:assert/strict';
import {parseHoldings,readSample,initialState,isPending,universe,type Decision} from '../src/lib/sample-research';
test('CSV distinguishes supported, duplicate, malformed and unsupported holdings',()=>{
 const rows=parseHoldings('\uFEFFsymbol,quantity,notes\r\nINFY,12,"long, term"\r\nINFY,4,duplicate\r\nUNKNOWN,20,no\r\nTCS,-1,no\r\nITC,,no');
 assert.equal(rows[0].reason,'');assert.equal(rows[0].quantity,12);
 assert.equal(rows[1].reason,'Duplicate symbol');assert.equal(rows[2].reason,'Outside initial coverage');assert.equal(rows[3].reason,'Invalid quantity');assert.equal(rows[4].reason,'Invalid quantity');
 assert.throws(()=>parseHoldings('ticker,quantity\nINFY,1'),/headers/);
 assert.throws(()=>parseHoldings('symbol,quantity\n"INFY,1'),/quote/);
 assert.throws(()=>parseHoldings('symbol,quantity\n'+'INFY,1\n'.repeat(101)),/100/);
});
test('review decisions survive serialization and deferred reviews return when due',()=>{
 const decision:Decision={id:'d1',reviewId:'infy-deals',company:'Infosys',action:'Revisit later',before:'old',after:'old',note:'wait',date:'2026-11-02T00:00:00Z',evidence:'snapshot',deferUntil:'2026-11-09T00:00:00Z'};
 const state=readSample(JSON.stringify({...initialState,decisions:[decision]}));
 assert.equal(isPending(state,'infy-deals',Date.parse('2026-11-03')),false);
 assert.equal(isPending(state,'infy-deals',Date.parse('2026-11-09')),true);
 assert.equal(isPending({...state,decisions:[{...decision,id:'d2',action:'Edited assumption'},decision]},'infy-deals',Date.parse('2026-11-03')),false);
 assert.equal(state.decisions[0].evidence,'snapshot');
});
test('invalid persisted data is rejected and universe contains the locked 25 companies',()=>{
 assert.equal(universe.length,25);assert.equal(new Set(universe.map(c=>c.symbol)).size,25);
 assert.throws(()=>readSample(JSON.stringify({...initialState,following:['UNKNOWN']})));
 assert.throws(()=>readSample(JSON.stringify({...initialState,decisions:[{}]})));
 assert.deepEqual(readSample(JSON.stringify(initialState)),initialState);
});
