import { requestIdentity } from '@/lib/auth';
import { exportJournal } from '@/lib/journal';
import { journalFile,journalFormat,JournalLimitError } from '@/lib/journal-format';
export const dynamic='force-dynamic';
export const runtime='nodejs';
const headers={'Cache-Control':'private, no-store','Vary':'Authorization','X-Content-Type-Options':'nosniff'};
export async function GET(request:Request){
 const owner=await requestIdentity(request);
 if(!owner)return Response.json({error:'Sign in with an invited account.'},{status:401,headers});
 let format;try{format=journalFormat(new URL(request.url).searchParams.get('format'));}catch{return Response.json({error:'Choose JSON or Markdown.'},{status:400,headers});}
 try{
  const file=journalFile(await exportJournal(owner),format);
  // Stream bounded chunks so hosted exports do not hit the buffered-response limit.
  const bytes=new TextEncoder().encode(file.content);let offset=0;
  const stream=new ReadableStream<Uint8Array>({pull(controller){if(offset>=bytes.length){controller.close();return;}controller.enqueue(bytes.subarray(offset,offset+64*1024));offset+=64*1024;}});
  return new Response(stream,{headers:{...headers,'Content-Type':file.mediaType,'Content-Disposition':`attachment; filename="${file.filename}"`}});
 }catch(e){return Response.json({error:e instanceof JournalLimitError?e.message:'Your journal could not be exported. Please try again.'},{status:e instanceof JournalLimitError?413:503,headers});}
}
