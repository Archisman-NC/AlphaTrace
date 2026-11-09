import test from 'node:test';
import assert from 'node:assert/strict';
import { validQuantity,validateMutation,readHoldingsCsv,previewHoldings } from '../src/lib/portfolio-input';
import { invitedUser,publicAuthConfig } from '../src/lib/invites';
import { boundedJson,RequestBodyError } from '../src/lib/http-input';
test('quantities remain exact and malformed/zero/non-finite values are rejected',()=>{
 assert.equal(validQuantity('000120.340000'),'120.34');
 assert.equal(validQuantity('999999999999.999999'),'999999999999.999999');
 for(const value of ['0','0.000000','-1','NaN','Infinity','1e6','1,200','1.1234567','1000000000000','',42])assert.throws(()=>validQuantity(value));
});
test('mutations derive no ownership from client data and validate import shape',()=>{
 const result=validateMutation({version:0,owner_id:'someone-else',command:{type:'holding',symbol:' infy ',quantity:'12.500000',owner_id:'someone-else'}});
 assert.deepEqual(result,{version:0,command:{type:'holding',symbol:'INFY',quantity:'12.5'}});
 assert.throws(()=>validateMutation({version:-1,command:{type:'follow',symbol:'INFY',followed:true}}));
 assert.throws(()=>validateMutation({version:0,command:{type:'follow',symbol:'UNKNOWN',followed:true}}));
 assert.throws(()=>validateMutation({version:0,command:{type:'import',holdings:[{symbol:'INFY',quantity:'1'},{symbol:'infy',quantity:'2'}]}}),/duplicate/i);
 assert.throws(()=>validateMutation({version:0,command:{type:'import',holdings:[]}}));
});
test('CSV maps broker columns and excludes duplicate, unsupported and invalid rows',()=>{
 const table=readHoldingsCsv('\uFEFFNote,Trading Symbol,Qty\r\n"long, term",INFY,120.5\r\nnote,TCS,2\r\nnote,TCS,4\r\nnote,OTHER,1\r\nnote,ITC,-1');
 const rows=previewHoldings(table,1,2);
 assert.equal(rows[0].quantity,'120.5');assert.equal(rows[0].error,'');
 assert.match(rows[1].error,/Duplicate/);assert.match(rows[2].error,/Duplicate/);
 assert.match(rows[3].error,/outside/);assert.match(rows[4].error,/positive/);
 assert.throws(()=>previewHoldings(table,1,1),/different/);
 assert.throws(()=>readHoldingsCsv('symbol,qty\nINFY,2,extra'),/same columns/);
 assert.throws(()=>readHoldingsCsv('symbol,symbol\nINFY,2'),/unique/);
 assert.throws(()=>readHoldingsCsv('symbol,qty\n"INFY"oops,2'),/after a quoted/);
 assert.throws(()=>readHoldingsCsv('symbol,qty\n'+('INFY,1\n'.repeat(101))),/100/);
 assert.throws(()=>readHoldingsCsv('x'.repeat(1024*1024+1)),/1 MB/);
});
test('invitation gate denies absent, malformed or oversized configuration',()=>{
 const id='11111111-1111-4111-8111-111111111111';
 assert.equal(invitedUser(id,undefined),false);
 assert.equal(invitedUser(id,id),true);
 assert.equal(invitedUser(id,`${id},broken`),false);
 assert.equal(invitedUser(id,Array(6).fill(id).join(',')),false);
 assert.equal(invitedUser('22222222-2222-4222-8222-222222222222',id),false);
});
test('browser auth config never exposes privileged keys',()=>{
 const config={SUPABASE_URL:'https://example.supabase.co',SUPABASE_PUBLISHABLE_KEY:'sb_publishable_example'};
 assert.deepEqual(publicAuthConfig(config),{url:config.SUPABASE_URL,key:config.SUPABASE_PUBLISHABLE_KEY});
 for(const role of ['service_role','authenticated'])assert.equal(publicAuthConfig({...config,SUPABASE_PUBLISHABLE_KEY:`header.${Buffer.from(JSON.stringify({role})).toString('base64url')}.sig`}),null);
 assert.equal(publicAuthConfig({...config,SUPABASE_PUBLISHABLE_KEY:'sb_secret_private'}),null);
 assert.equal(publicAuthConfig({...config,SUPABASE_URL:'http://example.com'}),null);
});
test('bounded body reader rejects wrong content, invalid JSON and excess streamed bytes',async()=>{
 const req=(body:string,type='application/json')=>new Request('http://localhost/api/portfolio',{method:'POST',headers:{'content-type':type},body});
 assert.deepEqual(await boundedJson(req('{"version":0}')),{version:0});
 await assert.rejects(boundedJson(req('{}','text/plain')),e=>e instanceof RequestBodyError&&e.status===415);
 await assert.rejects(boundedJson(req('{')),e=>e instanceof RequestBodyError&&e.status===400);
 await assert.rejects(boundedJson(req('x'.repeat(50)),20),e=>e instanceof RequestBodyError&&e.status===413);
});
