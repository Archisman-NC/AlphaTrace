import assert from 'node:assert/strict';
const base=process.env.PORTFOLIO_TEST_URL??'http://127.0.0.1:3000';
if(!/^http:\/\/(127\.0\.0\.1|localhost):\d+$/.test(base))throw Error('This check only runs against a loopback test server.');
const token=process.env.ALPHATRACE_DEV_TOKEN;
if(!token)throw Error('Configure the local test token.');
const headers={Authorization:`Bearer ${token}`,'Content-Type':'application/json'};
assert.equal((await fetch(base+'/api/portfolio')).status,401);
assert.equal((await fetch(base+'/api/portfolio',{headers:{Authorization:'Bearer invalid'}})).status,401);
if(process.env.TEST_PRODUCTION==='true'){
 assert.equal((await fetch(base+'/api/portfolio',{headers})).status,401,'production rejects local identity');
 const html=await (await fetch(base+'/workspace')).text();
 assert.ok(!html.includes('Local development identity'),'production HTML must not render a development identity');
 assert.ok(html.includes('Account sign-in is not configured'),'unconfigured production must show closed auth state');
 console.log('PASS: production denies local tokens and renders no private development portfolio');
}else{
 const get=await fetch(base+'/api/portfolio',{headers});assert.equal(get.status,200);assert.equal(get.headers.get('cache-control'),'no-store');
 const {portfolio}=await get.json();assert.equal(typeof portfolio.version,'number');
 assert.ok(portfolio.entries.every((e:Record<string,unknown>)=>!('owner_id' in e)));
 const post=(body:string,type='application/json')=>fetch(base+'/api/portfolio',{method:'POST',headers:{...headers,'Content-Type':type},body});
 assert.equal((await post('{}','text/plain')).status,415);
 assert.equal((await post('{')).status,400);
 assert.equal((await post('x'.repeat(32769))).status,413);
 assert.equal((await post(JSON.stringify({version:portfolio.version,command:{type:'holding',symbol:'INFY',quantity:'-1'}}))).status,400);
 assert.equal((await post(JSON.stringify({version:portfolio.version+1,command:{type:'holding',symbol:'INFY',quantity:'1'},owner_id:'22222222-2222-4222-8222-222222222222'}))).status,409);
 const after=await (await fetch(base+'/api/portfolio',{headers})).json();assert.deepEqual(after.portfolio,portfolio,'rejected requests must not mutate any holdings');
 console.log('PASS: API auth, no-store, media type, JSON validation, bounded body, optimistic conflict and no rejected-write mutation');
}
