import {createClient} from '@supabase/supabase-js';
import {publicAuthConfig} from '@/lib/invites';
import {requestIdentity} from '@/lib/auth';
import {exportAccount} from '@/lib/account-export';
import {JournalLimitError} from '@/lib/journal-format';
export const runtime='nodejs';export const dynamic='force-dynamic';
const headers={'Cache-Control':'private, no-store','Vary':'Authorization','X-Content-Type-Options':'nosniff'};
export async function GET(request:Request){
 const owner=await requestIdentity(request);if(!owner)return Response.json({error:'Your session expired. Sign in again.'},{status:401,headers});
 try{const data=await exportAccount(owner),config=publicAuthConfig(process.env);
  if(!config)throw Error('Account identity unavailable');
  const {data:identity,error}=await createClient(config.url,config.key,{auth:{persistSession:false,autoRefreshToken:false}}).auth.getUser(request.headers.get('authorization')!.slice(7));
  if(error||identity.user?.id!==owner)throw Error('Account identity unavailable');
  data.account={email:identity.user.email,createdAt:identity.user.created_at,emailConfirmedAt:identity.user.email_confirmed_at};
  const bytes=new TextEncoder().encode(JSON.stringify(data,null,2)+'\n');if(bytes.length>20*1024*1024)throw new JournalLimitError('The export exceeds the download limit. Contact the beta operator for a full export.');let offset=0;const body=new ReadableStream<Uint8Array>({pull(c){if(offset>=bytes.length){c.close();return;}c.enqueue(bytes.subarray(offset,offset+65536));offset+=65536;}});return new Response(body,{headers:{...headers,'Content-Type':'application/json; charset=utf-8','Content-Disposition':`attachment; filename="alphatrace-account-${data.exportedAt.slice(0,10)}.json"`}});}catch(e){return Response.json({error:e instanceof JournalLimitError?e.message:'Account data could not be exported. No partial file was created.'},{status:e instanceof JournalLimitError?413:503,headers});}
}
