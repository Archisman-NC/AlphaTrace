import { timingSafeEqual } from 'node:crypto';
import { UUID } from '@/lib/access';
import { ingestResearch } from '@/lib/research';
import { boundedJson } from '@/lib/http-input';
import { researchJson as json,researchError } from '@/lib/research-response';
export const runtime='nodejs';
export const dynamic='force-dynamic';
// This credential can submit drafts/checks only. Review requires a real operator session.
export async function POST(request:Request){
 const expected=process.env.ALPHATRACE_INGEST_TOKEN??'',actual=request.headers.get('authorization')?.replace(/^Bearer /,'')??'',id=process.env.ALPHATRACE_INGEST_USER_ID;
 const a=Buffer.from(actual),b=Buffer.from(expected);
 if(b.length<32||a.length!==b.length||!timingSafeEqual(a,b)||!id||!UUID.test(id))return json({error:'Invalid ingestion credential.'},401);
 try{return json(await ingestResearch(id,await boundedJson(request,131072)));}
 catch(e){const result=researchError(e);return json(result,result.status);}
}
