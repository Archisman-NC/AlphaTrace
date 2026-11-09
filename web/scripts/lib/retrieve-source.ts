import { spawn } from 'node:child_process';
import { mkdtemp,readFile,stat,rm,access } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { MonitorSymbol } from '../../src/lib/monitor-contract';

export const sourceFailure=(symbol:MonitorSymbol,code:string,message:string)=>({failureCode:code,check:{symbol,checkedAt:new Date().toISOString(),status:'failed',message,fingerprints:[]}});
export async function checkSourceTools(directory:string) {
 await access(join(directory,'batch.py'));await access(join(directory,'sec_infy.py'));await access(join(directory,'sources.json'));
 for(const command of ['python3','pdfinfo','pdftotext'])await new Promise<void>((resolve,reject)=>{
  const child=spawn(command,[command==='python3'?'--version':'-v'],{stdio:'ignore',timeout:10_000,env:{PATH:process.env.PATH,NODE_ENV:process.env.NODE_ENV}});
  child.once('error',()=>reject(Error('Source processing dependency unavailable.')));
  child.once('close',code=>code===0?resolve():reject(Error('Source processing dependency unavailable.')));
 });
}
export async function retrieveSource(symbol:MonitorSymbol,scripts:string) {
 const directory=await mkdtemp(join(tmpdir(),'alphatrace-cloud-'));
 try {
  const result=await new Promise<{code:number|null;timeout:boolean;failed:boolean}>(resolve=>{
   let timeout=false;
   const child=spawn('python3',[join(scripts,'batch.py'),'--symbol',symbol,'--output-dir',directory],{
    cwd:directory,env:{PATH:process.env.PATH,NODE_ENV:process.env.NODE_ENV,LANG:'C.UTF-8',ALPHATRACE_RESEARCH_ROOT:directory},stdio:'ignore'});
   const timer=setTimeout(()=>{timeout=true;child.kill('SIGKILL');},180_000);
   child.once('error',()=>{clearTimeout(timer);resolve({code:null,timeout:false,failed:true});});
   child.once('close',code=>{clearTimeout(timer);resolve({code,timeout,failed:false});});
  });
  if(result.timeout)return sourceFailure(symbol,'timeout','The source check exceeded its three-minute limit. No fresh evidence was confirmed.');
  if(result.failed)return sourceFailure(symbol,'dependency','The source reader could not start. Operator attention is required.');
  const path=join(directory,symbol+'.json');
  if((await stat(path)).size>131072)throw Error();
  const bundle=JSON.parse(await readFile(path,'utf8'));
  if(bundle.check?.symbol!==symbol||!['success','failed'].includes(bundle.check.status)||result.code!==0&&bundle.check.status==='success')throw Error();
  return bundle;
 }catch{return sourceFailure(symbol,'validation','Source output could not be validated. Manual source review is required.');}
 finally{await rm(directory,{recursive:true,force:true});}
}
