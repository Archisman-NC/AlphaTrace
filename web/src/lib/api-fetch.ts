/** Application API requests only. No automatic write retry or session clearing. */
export async function apiFetch(input:RequestInfo|URL,init?:RequestInit):Promise<Response>{
 const authenticated=new Headers(init?.headers).has('Authorization');
 try{
  const response=await fetch(input,{...init,...(!init?.signal&&(!init?.method||init.method==='GET')?{signal:AbortSignal.timeout(20000)}:{})});
  if(authenticated&&typeof window!=='undefined'&&(response.status===401||response.status>=500))window.dispatchEvent(new CustomEvent('alphatrace:request-status',{detail:response.status===401?'session':'service'}));
  const json=response.json.bind(response);
  response.json=async()=>{try{return await json();}catch{throw new TypeError('The server response could not be read. A submitted change may have completed; check its status or retry the preserved request.');}};
  return response;
 }catch{
  if(typeof window!=='undefined')window.dispatchEvent(new CustomEvent('alphatrace:request-status',{detail:'connection'}));
  // Existing mutation forms classify TypeError as an unconfirmed transport result
  // and preserve their command UUID. Abort/timeout must follow that same path.
  throw new TypeError('The connection was interrupted. A submitted change may have completed; check its status or retry the preserved request.');
 }
}
