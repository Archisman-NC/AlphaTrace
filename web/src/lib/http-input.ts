export class RequestBodyError extends Error {
 constructor(message:string,public status:number){super(message);}
}
export async function boundedJson(request:Request,limit=32768):Promise<unknown> {
 if(request.headers.get('content-type')?.split(';')[0].trim()!=='application/json')throw new RequestBodyError('JSON required.',415);
 const reader=request.body?.getReader();
 if(!reader)throw new RequestBodyError('Request body is required.',400);
 const chunks:Uint8Array[]=[];let size=0;
 try{
  while(true){const part=await reader.read();if(part.done)break;size+=part.value.byteLength;if(size>limit){await reader.cancel();throw new RequestBodyError('Request body is too large.',413);}chunks.push(part.value);}
  return JSON.parse(Buffer.concat(chunks).toString('utf8'));
 }catch(e){if(e instanceof RequestBodyError)throw e;throw new RequestBodyError('Invalid JSON.',400);}
 finally{reader.releaseLock();}
}
