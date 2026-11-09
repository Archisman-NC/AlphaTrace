export type WorkerMode='local'|'cloud';
export type WorkerStatus={worker_seen_at:string|null;worker_mode?:WorkerMode};
// Free-plan hourly dispatch, with scheduling window and tolerance for one miss.
export const cloudHeartbeatMs=150*60_000;
export function workerOnline(target:WorkerStatus,now=Date.now(),localGraceMs=180_000){
 return !!target.worker_seen_at&&now-Date.parse(target.worker_seen_at)<=(target.worker_mode==='cloud'?cloudHeartbeatMs:localGraceMs);
}
export function workerLabel(targets:WorkerStatus[],now=Date.now(),localGraceMs=180_000){
 const cloud=targets.some(t=>t.worker_mode==='cloud');
 const online=targets.some(t=>(t.worker_mode==='cloud')===cloud&&workerOnline(t,now,localGraceMs));
 return cloud?(online?'Cloud scheduler checked in':'Cloud scheduler overdue'):(online?'Local runner online':'Local runner offline');
}
