import test from 'node:test';
import assert from 'node:assert/strict';
import {createAutoRefresh,inboxHasChanges,type RefreshState} from '../src/lib/auto-refresh';
import type {InboxList,InboxItem} from '../src/lib/inbox-contract';

function harness(run:()=>Promise<boolean|null>){
 let time=0,available=true,id=0;
 const timers=new Map<number,{fn:()=>void;at:number}>(),states:RefreshState[]=[];
 const loop=createAutoRefresh({run,now:()=>time,available:()=>available,changed:s=>states.push(s),schedule:(fn,delay)=>{timers.set(++id,{fn,at:time+delay});return id;},cancel:timer=>{timers.delete(timer as number);}});
 return {loop,states,timers,setAvailable:(v:boolean)=>{available=v;},advance:async(ms:number)=>{
  time+=ms;for(const [key,timer] of [...timers])if(timer.at<=time){timers.delete(key);timer.fn();}
  await new Promise(resolve=>setImmediate(resolve));
 }};
}
test('foreground refresh pauses when hidden/offline and catches up on return without an event storm',async()=>{
 let calls=0;const h=harness(async()=>{calls++;return true;});
 h.setAvailable(false);await h.advance(60_000);assert.equal(calls,0);
 h.setAvailable(true);await h.loop.wake();assert.equal(calls,1);
 await h.loop.wake();assert.equal(calls,1);
 await h.advance(60_000);assert.equal(calls,2);assert.equal(h.states.at(-1)?.failed,false);
 h.loop.stop();await h.advance(600_000);assert.equal(calls,2);
});
test('slow responses do not overlap and stopping ignores an in-flight completion',async()=>{
 let complete!:(value:boolean)=>void,calls=0;
 const h=harness(()=>{calls++;return new Promise<boolean>(resolve=>{complete=resolve;});});
 await h.advance(60_000);await h.loop.wake();await h.advance(60_000);assert.equal(calls,1);
 const emitted=h.states.length;h.loop.stop();complete(true);await h.advance(1);
 assert.equal(h.states.length,emitted);assert.equal(h.timers.size,0);
});
test('failed checks retain the last success and back off even on focus/online events',async()=>{
 let success=true,calls=0;const h=harness(async()=>{calls++;return success;});
 await h.advance(60_000);assert.equal(h.states.at(-1)?.checkedAt,60_000);
 success=false;await h.advance(60_000);assert.equal(calls,2);
 assert.deepEqual(h.states.at(-1),{checking:false,failed:true,checkedAt:60_000});
 await h.advance(60_000);await h.loop.wake();assert.equal(calls,2);
 success=true;await h.advance(60_000);assert.equal(calls,3);assert.equal(h.states.at(-1)?.failed,false);
 h.loop.stop();
});
test('a superseded read does not report an outage or claim a successful new check',async()=>{
 const h=harness(async()=>null);await h.advance(60_000);
 assert.deepEqual(h.states.at(-1),{checking:false,failed:false,checkedAt:null});
 assert.equal(h.timers.size,1);h.loop.stop();
});
const item=(id:string):InboxItem=>({update_id:id,symbol:'WIPRO',title:'Fixture',publication_version:1,period_end:'2026-09-30',created_at:'2026-11-08T10:00:00Z',version:0,state:'unreviewed',revisit_at:null,match_count:0,withdrawn_at:null,withdrawal_reason:null,newer_update_id:null});
const list=(items:InboxItem[]):InboxList=>({items,nextCursor:null,counts:{pending:items.length,deferred:0,reviewed:0,all:items.length}});
test('new publications, withdrawals, corrections, other-tab responses and due reviews flag changes',()=>{
 const current=list([item('a')]),copy=structuredClone(current);
 assert.equal(inboxHasChanges(current,copy),false);
 assert.equal(inboxHasChanges(current,list([item('b'),item('a')])),true);
 for(const patch of [{withdrawn_at:'2026-11-08T11:00:00Z',withdrawal_reason:'Wrong period'},{newer_update_id:'b'},{version:1,state:'kept' as const}]){
  const fresh=structuredClone(current);Object.assign(fresh.items[0],patch);assert.equal(inboxHasChanges(current,fresh),true);
 }
 const due=structuredClone(current);due.counts.pending++;due.counts.deferred--;assert.equal(inboxHasChanges(current,due),true);
 assert.deepEqual(current,copy,'checking does not mutate the open review');
});
test('loading older pages alone does not produce a false update notification',()=>{
 const current=list(Array.from({length:30},(_,i)=>item(String(i))));
 const first={...current,items:current.items.slice(0,25),nextCursor:'next'};
 assert.equal(inboxHasChanges(current,first),false);
});
