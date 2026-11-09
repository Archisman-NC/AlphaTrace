import { Webhook } from 'svix';
import { RequestBodyError } from './http-input';
export async function verifyEmailWebhook(request:Request,secret:string){
 const reader=request.body?.getReader();if(!reader)throw new RequestBodyError('Body required.',400);
 const chunks:Uint8Array[]=[];let size=0;
 try{while(true){const part=await reader.read();if(part.done)break;size+=part.value.byteLength;if(size>65536){await reader.cancel();throw new RequestBodyError('Body too large.',413);}chunks.push(part.value);}}finally{reader.releaseLock();}
 const id=request.headers.get('svix-id')??'',timestamp=request.headers.get('svix-timestamp')??'',signature=request.headers.get('svix-signature')??'';
 if(id.length>200||signature.length>2048)throw new RequestBodyError('Invalid webhook.',400);
 let event:unknown;
 try{const raw=Buffer.concat(chunks).toString('utf8');new Webhook(secret).verify(raw,{'svix-id':id,'svix-timestamp':timestamp,'svix-signature':signature});event=JSON.parse(raw);}catch{throw new RequestBodyError('Invalid webhook signature.',400);}
 if(!event||typeof event!=='object')throw new RequestBodyError('Invalid webhook event.',400);
 const e=event as {type?:unknown;created_at?:unknown;data?:{email_id?:unknown}};
 if(!id||typeof e.type!=='string'||e.type.length>80||typeof e.created_at!=='string'||e.created_at.length>80||!Number.isFinite(Date.parse(e.created_at))||Date.parse(e.created_at)>Date.now()+300000||typeof e.data?.email_id!=='string'||!e.data.email_id||e.data.email_id.length>200)throw new RequestBodyError('Invalid webhook event.',400);
 return {id,type:e.type,at:e.created_at,providerId:e.data.email_id};
}
