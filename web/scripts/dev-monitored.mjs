import { spawn } from 'node:child_process';
// Use the project-pinned Node. Both processes inherit the same local configuration.
if(!process.env.ALPHATRACE_INGEST_USER_ID)throw Error('Configure the enrolled research operator before using dev:monitored.');
const processes=[
 spawn(process.execPath,['node_modules/next/dist/bin/next','dev','--hostname','127.0.0.1'],{stdio:'inherit'}),
 ...(process.env.ALPHATRACE_CLOUD_MONITORING==='true'?[]:[spawn(process.execPath,['--import','tsx','scripts/research-monitor.ts','--watch'],{stdio:'inherit'})]),
];
if(process.env.ALPHATRACE_CLOUD_MONITORING!=='true'&&process.env.INDIANAPI_API_KEY)
 processes.push(spawn(process.execPath,['--import','tsx','scripts/indianapi-worker.ts','--watch'],{stdio:'inherit'}));
let stopping=false;
function stop(code=0){
 if(stopping)return;stopping=true;process.exitCode=code;
 for(const child of processes)child.kill('SIGTERM');
 const timer=setTimeout(()=>{for(const child of processes)child.kill('SIGKILL');},5000);timer.unref();
}
process.on('SIGINT',()=>stop());process.on('SIGTERM',()=>stop());
for(const child of processes){
 child.on('error',()=>{console.error('A local service could not start.');stop(1);});
 child.on('exit',code=>{if(!stopping)stop(code??1);});
}
