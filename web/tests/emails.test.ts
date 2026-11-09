import { test } from 'node:test';
import assert from 'node:assert/strict';
import { randomBytes } from 'node:crypto';
import { Webhook } from 'svix';
import { digestPayload,emailConfig,emailsAvailable,validateEmailPreferences,type DigestItem } from '../src/lib/email-contract';
import { sendResend } from '../src/lib/resend';
import { verifyEmailWebhook } from '../src/lib/email-webhook';
import { emailEnvironment } from '../scripts/lib/email-environment';
import { confirmedRecipient } from '../src/lib/email-recipient';
const item:DigestItem={key:'publication:abc',updateId:'abc',symbol:'TCS',title:'<script>alert("x")</script> & earnings',periodEnd:'2026-09-30',kind:'publication',reviewVersion:null};
const config={from:'AlphaTrace <updates@example.test>',origin:'https://alpha.example.test',apiKey:'re_test_private'};
test('only the same confirmed, active account can receive a digest',()=>{
 const user={id:'owner',email:'pilot@example.test',email_confirmed_at:new Date().toISOString()};
 assert.equal(confirmedRecipient('owner',user),user.email);
 assert.equal(confirmedRecipient('other',user),null);
 assert.equal(confirmedRecipient('owner',{...user,email_confirmed_at:undefined}),null);
 assert.equal(confirmedRecipient('owner',{...user,deleted_at:new Date().toISOString()}),null);
 assert.equal(confirmedRecipient('owner',{...user,banned_until:new Date(Date.now()+86400000).toISOString()}),null);
});
test('digest escapes source text, uses exact review links, supports unsubscribe, omits private investment fields',()=>{
 const p=digestPayload(config,'pilot@example.test',[item],'x'.repeat(43));
 assert.ok(!p.html.includes('<script>'));assert.match(p.html,/&lt;script&gt;/);assert.match(p.html,/inbox\?filter=all&amp;update=abc/);
 assert.match(p.headers['List-Unsubscribe'],/\/api\/email\/unsubscribe\?token=/);assert.equal(p.headers['List-Unsubscribe-Post'],'List-Unsubscribe=One-Click');
 assert.ok(!JSON.stringify(p).includes(config.apiKey));assert.match(p.text,/Opening an update does not mark it reviewed/);
 assert.throws(()=>digestPayload(config,'pilot@example.test',[],'token'));
});
test('opt-in requires explicit consent, proper version and complete verified-domain runtime configuration',()=>{
 assert.throws(()=>validateEmailPreferences({version:0,enabled:true,reminders:true}),/Confirm/);
 assert.deepEqual(validateEmailPreferences({version:1,enabled:false,reminders:true}),{version:1,enabled:false,reminders:true});
 assert.throws(()=>validateEmailPreferences({version:-1,enabled:false,reminders:true}),/Reload/);
 const env={RESEND_API_KEY:'re_test',RESEND_FROM_EMAIL:'updates@example.test',ALPHATRACE_APP_URL:'https://alpha.example.test',RESEND_WEBHOOK_SECRET:'whsec_test',SUPABASE_SECRET_KEY:'sb_secret_test',ALPHATRACE_EMAILS_ENABLED:'true',ALPHATRACE_INVITED_USER_IDS:'11111111-1111-4111-8111-111111111111'};
 assert.equal(emailsAvailable(env),true);assert.equal(emailsAvailable({...env,RESEND_FROM_EMAIL:'onboarding@resend.dev'}),false);assert.equal(emailsAvailable({...env,RESEND_WEBHOOK_SECRET:''}),false);
 assert.throws(()=>emailConfig({...env,ALPHATRACE_APP_URL:'http://example.test'}));assert.throws(()=>emailConfig({...env,RESEND_FROM_EMAIL:'a@example.test\nBcc: victim@example.test'}));
 assert.deepEqual(emailEnvironment({RESEND_API_KEY:'secret'}),{ALPHATRACE_EMAILS_ENABLED:'false'});
 assert.ok(!('SUPABASE_DB_PASSWORD' in emailEnvironment({...env,SUPABASE_DB_PASSWORD:'secret'})));
});
test('provider sends use stable idempotency keys, do not follow redirects, and redact reflected secrets on failures',async()=>{
 const payload=digestPayload(config,'pilot@example.test',[item],'x'.repeat(43));let captured:RequestInit|undefined;
 const fake:typeof fetch=async(url,options)=>{assert.equal(url,'https://api.resend.com/emails');captured=options;return Response.json({id:'provider-id'});};
 assert.deepEqual(await sendResend(payload,config.apiKey,'delivery-1',fake),{state:'accepted',providerId:'provider-id'});
 assert.equal(new Headers(captured?.headers).get('Idempotency-Key'),'alphatrace-digest/delivery-1');assert.equal(captured?.redirect,'error');assert.equal(captured?.body,JSON.stringify(payload));
 for(const code of [429,500,403]){const r=await sendResend(payload,config.apiKey,'delivery-1',async()=>Response.json({message:config.apiKey},{status:code}));assert.equal(r.state,code===403?'failed':'retry');assert.ok(!JSON.stringify(r).includes(config.apiKey));}
 assert.equal((await sendResend(payload,config.apiKey,'delivery-1',async()=>{throw Error(config.apiKey);})).state,'retry');
});
test('webhooks require authentic, unmodified, recent signed raw bodies',async()=>{
 const secret='whsec_'+randomBytes(32).toString('base64'),hook=new Webhook(secret),id='msg_test',now=new Date(),body=JSON.stringify({type:'email.delivered',created_at:now.toISOString(),data:{email_id:'email_123'}});
 const request=(raw=body,at=now)=>new Request('https://alpha.example.test/api/email/webhook',{method:'POST',headers:{'svix-id':id,'svix-timestamp':String(Math.floor(at.getTime()/1000)),'svix-signature':hook.sign(id,at,body)},body:raw});
 assert.deepEqual(await verifyEmailWebhook(request(),secret),{id,type:'email.delivered',at:now.toISOString(),providerId:'email_123'});
 await assert.rejects(verifyEmailWebhook(request(body+' '),secret),/signature/);
 await assert.rejects(verifyEmailWebhook(request(body,new Date(Date.now()-10*60*1000)),secret),/signature/);
 await assert.rejects(verifyEmailWebhook(request(), 'whsec_'+randomBytes(32).toString('base64')),/signature/);
});
