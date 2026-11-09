import { verifyEmailWebhook } from '@/lib/email-webhook';
import { recordEmailEvent } from '@/lib/emails';
import { RequestBodyError } from '@/lib/http-input';
export const runtime='nodejs';
export const dynamic='force-dynamic';
export async function POST(request:Request){
 const headers={'Cache-Control':'no-store'},secret=process.env.RESEND_WEBHOOK_SECRET;
 if(!secret)return Response.json({error:'Webhook is not configured.'},{status:503,headers});
 try{
  const event=await verifyEmailWebhook(request,secret);
  const recorded=await recordEmailEvent(event.id,event.providerId,event.type,event.at);
  // Resend may notify us before the send transaction has recorded its message ID.
  return new Response(null,{status:recorded?200:503,headers});
 }catch(e){return Response.json({error:e instanceof RequestBodyError?e.message:'Webhook processing is temporarily unavailable.'},{status:e instanceof RequestBodyError?e.status:503,headers});}
}
