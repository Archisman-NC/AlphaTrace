/** Bounded foreground refresh, independent of React so timing/races can be tested. */
import type {InboxList} from './inbox-contract';
export const refreshIntervalMs = 60_000;
export function inboxHasChanges(current:InboxList|null,fresh:InboxList):boolean {
 return !current||JSON.stringify(current.counts)!==JSON.stringify(fresh.counts)||JSON.stringify(current.items.slice(0,25))!==JSON.stringify(fresh.items);
}
export type RefreshState = {checking:boolean; failed:boolean; checkedAt:number|null};
type Options = {
 run:()=>Promise<boolean|null>;
 available:()=>boolean;
 changed:(state:RefreshState)=>void;
 now?:()=>number;
 schedule?:(fn:()=>void,delay:number)=>unknown;
 cancel?:(timer:unknown)=>void;
};
export function createAutoRefresh(options:Options) {
 const now=options.now??Date.now;
 const schedule=options.schedule??((fn,delay)=>setTimeout(fn,delay));
 const cancel=options.cancel??(timer=>clearTimeout(timer as ReturnType<typeof setTimeout>));
 let stopped=false,running=false,failures=0,lastAttempt=-Infinity,timer:unknown;
 let state:RefreshState={checking:false,failed:false,checkedAt:null};
 const emit=(patch:Partial<RefreshState>)=>{state={...state,...patch};if(!stopped)options.changed(state);};
 const delay=()=>Math.min(refreshIntervalMs*2**failures,300_000);
 function plan(){if(!stopped)timer=schedule(()=>{void check();},delay());}
 async function check(wake=false){
  if(stopped||running)return;
  // Focus/online events cannot evade the failure backoff or hammer the server.
  if(wake&&now()-lastAttempt<(failures?delay():10_000))return;
  cancel(timer);
  if(!options.available()){plan();return;}
  running=true;lastAttempt=now();emit({checking:true});
  let success:boolean|null=false;
  try{success=await options.run();}catch{/* Keep the last confirmed snapshot. */}
  if(stopped)return;
  // A manual refresh/navigation superseded this read; it is not an outage.
  if(success===null){running=false;emit({checking:false});plan();return;}
  running=false;failures=success?0:Math.min(failures+1,3);
  emit({checking:false,failed:!success,...(success?{checkedAt:now()}:{})});plan();
 }
 plan();
 return {wake:()=>check(true),stop:()=>{stopped=true;cancel(timer);}};
}
