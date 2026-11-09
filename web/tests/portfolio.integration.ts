import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import pg from 'pg';
import { databaseConfig } from '../src/lib/database-config.mjs';
import { asUser,closeDatabase } from '../src/lib/db';
import { getPortfolio,updatePortfolio } from '../src/lib/portfolios';
import { universe } from '../src/lib/companies';
import { PortfolioConflictError } from '../src/lib/portfolio-input';
const alice=randomUUID(),bob=randomUUID();
const admin=new pg.Client(databaseConfig(process.env.MIGRATION_DATABASE_URL));
await admin.connect();
try{
 assert.deepEqual(await getPortfolio(alice),{version:0,entries:[]});
 let a=await updatePortfolio(alice,{version:0,command:{type:'holding',symbol:'INFY',quantity:'120.123456'}});
 assert.equal(a.entries[0].quantity,'120.123456');
 a=await updatePortfolio(alice,{version:a.version,command:{type:'follow',symbol:'INFY',followed:true}});
 assert.equal(a.entries[0].followed,true);
 const imported={type:'import',holdings:[{symbol:'INFY',quantity:'75.5'},{symbol:'TCS',quantity:'25'}]};
 a=await updatePortfolio(alice,{version:a.version,command:imported});
 const first=a.entries;
 a=await updatePortfolio(alice,{version:a.version,command:imported});
 assert.deepEqual(a.entries,first,'repeat imports must set absolute quantities, not add');
 assert.equal(a.entries.find(e=>e.symbol==='INFY')?.followed,true,'imports preserve watchlists');
 await closeDatabase();
 assert.deepEqual(await getPortfolio(alice),a,'new pool must read persisted data');
 assert.deepEqual(await getPortfolio(bob),{version:0,entries:[]});
 await updatePortfolio(bob,{version:0,command:{type:'holding',symbol:'INFY',quantity:'2'}});
 for(const sql of [
  'SELECT * FROM alphatrace.portfolio_entries WHERE owner_id=$1',
  "UPDATE alphatrace.portfolio_entries SET quantity=900 WHERE owner_id=$1",
  'DELETE FROM alphatrace.portfolio_entries WHERE owner_id=$1',
 ])assert.equal(await asUser(bob,async c=>(await c.query(sql,[alice])).rowCount),0,'other owner cannot read/write/delete');
 await assert.rejects(asUser(bob,c=>c.query("INSERT INTO alphatrace.portfolio_entries(owner_id,symbol,quantity) VALUES ($1,'ITC',99)",[alice])),/row-level security/);
 const raw=new pg.Client(databaseConfig(process.env.DATABASE_URL));await raw.connect();
 try{assert.equal((await raw.query('SELECT * FROM alphatrace.portfolio_entries')).rowCount,0);}finally{await raw.end();}
 await assert.rejects(updatePortfolio(alice,{version:0,command:{type:'holding',symbol:'INFY',quantity:'999'}}),PortfolioConflictError);
 assert.deepEqual(await getPortfolio(alice),a,'conflicts must not mutate data');
 const concurrent=await Promise.allSettled([
  updatePortfolio(alice,{version:a.version,command:{type:'holding',symbol:'ITC',quantity:'10'}}),
  updatePortfolio(alice,{version:a.version,command:{type:'holding',symbol:'LT',quantity:'20'}}),
 ]);
 assert.equal(concurrent.filter(x=>x.status==='fulfilled').length,1,'only one same-version writer may win');
 assert.equal(concurrent.filter(x=>x.status==='rejected'&&x.reason instanceof PortfolioConflictError).length,1);
 a=await getPortfolio(alice);
 for(const c of universe.slice(0,20))a=await updatePortfolio(alice,{version:a.version,command:{type:'follow',symbol:c.symbol,followed:true}});
 assert.equal(a.entries.filter(e=>e.followed).length,20);
 await assert.rejects(updatePortfolio(alice,{version:a.version,command:{type:'follow',symbol:universe[20].symbol,followed:true}}),/20/);
 assert.deepEqual(await getPortfolio(alice),a,'limit failure must roll back version and entries');
 await assert.rejects(updatePortfolio(alice,{version:a.version,command:{type:'import',holdings:[{symbol:'INFY',quantity:'10'},{symbol:'UNKNOWN',quantity:'1'}]}}));
 assert.deepEqual(await getPortfolio(alice),a,'invalid mixed imports must be atomic');
 a=await updatePortfolio(alice,{version:a.version,command:{type:'remove_holding',symbol:'INFY'}});
 assert.deepEqual(a.entries.find(e=>e.symbol==='INFY'),{symbol:'INFY',quantity:null,followed:true});
 a=await updatePortfolio(alice,{version:a.version,command:{type:'follow',symbol:'INFY',followed:false}});
 assert.equal(a.entries.some(e=>e.symbol==='INFY'),false);
 assert.equal((await getPortfolio(bob)).entries[0].quantity,'2.000000');
 console.log('PASS: exact quantity persistence, repeat import, two-user RLS read/write/delete isolation, owner spoof rejection, missing identity, concurrent version conflicts, watchlist cap, atomic rollback and independent holdings/watchlists');
}finally{
 await closeDatabase();await admin.query('DELETE FROM alphatrace.workspaces WHERE owner_id=ANY($1::uuid[])',[[alice,bob]]);await admin.end();
}
