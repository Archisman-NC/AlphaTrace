export class ServiceUnavailableError extends Error {}
export function serviceBoundary<T extends (...args:any[])=>Promise<Response>>(handler:T):(...args:Parameters<T>)=>Promise<Response>{
 return (async(...args:Parameters<T>)=>{
  try{return await handler(...args);}catch(error){
   const id=crypto.randomUUID(),unavailable=error instanceof ServiceUnavailableError;
   // Never log thrown messages, request bodies, URLs, tokens or private content.
   console.error(JSON.stringify({event:'api_failure',reference:id,kind:unavailable?'service_unavailable':'unexpected'}));
   return Response.json({error:unavailable?'A required service is temporarily unavailable. Your session has not been cleared. Try again shortly.':'The request could not be completed. Check the saved result before repeating a change.',reference:id},{status:unavailable?503:500,headers:{'Cache-Control':'private, no-store','X-Request-ID':id,...(unavailable?{'Retry-After':'30'}:{})}});
  }
 });
}
