import {test} from 'node:test';import assert from 'node:assert/strict';import {randomUUID} from 'node:crypto';
import {activateAuth} from '../src/lib/beta-auth';
test('activation reconciles interrupted creation and never replaces an existing password or trusts another email',async()=>{
 const previous=globalThis.fetch,env={SUPABASE_URL:process.env.SUPABASE_URL,SUPABASE_SECRET_KEY:process.env.SUPABASE_SECRET_KEY,SUPABASE_PUBLISHABLE_KEY:process.env.SUPABASE_PUBLISHABLE_KEY};Object.assign(process.env,{SUPABASE_URL:'https://synthetic.supabase.co',SUPABASE_SECRET_KEY:'sb_secret_synthetic',SUPABASE_PUBLISHABLE_KEY:'sb_publishable_synthetic'});
 const invite={id:randomUUID(),userId:randomUUID(),email:'invite@example.test',existing:false};let user:any=null,creates=0,mode='ok',signins=0;
 const session=(u:any)=>({access_token:'eyJhbGciOiJIUzI1NiJ9.'+Buffer.from(JSON.stringify({sub:u.id,exp:Math.floor(Date.now()/1000)+3600})).toString('base64url')+'.signature',refresh_token:'synthetic-refresh',expires_in:3600,token_type:'bearer',user:u});
 globalThis.fetch=async(url,init)=>{
  const path=new URL(String(url)).pathname,method=init?.method??'GET';assert.notEqual(method,'PUT','existing passwords must never be changed by activation');
  if(path.endsWith('/admin/users')&&method==='POST'){creates++;const body=JSON.parse(String(init?.body));assert.equal(body.id,invite.userId);assert.equal(body.email,invite.email);assert.equal(body.email_confirm,true);user={id:body.id,email:body.email,email_confirmed_at:new Date().toISOString(),created_at:new Date().toISOString(),app_metadata:body.app_metadata};if(mode==='lost')throw new TypeError('Synthetic lost creation reply');return Response.json(user);}
  if(path.endsWith('/admin/users/'+invite.userId))return user?Response.json(user):Response.json({code:'user_not_found',msg:'Not found'},{status:404});
  if(path.endsWith('/token')){signins++;return mode==='wrong-password'?Response.json({code:'invalid_credentials',msg:'Invalid'},{status:400}):Response.json(session(user));}
  if(path.endsWith('/logout'))return new Response(null,{status:204});
  if(path.endsWith('/user'))return Response.json(mode==='wrong-email'?{...user,email:'other@example.test'}:user);
  throw Error('Unexpected Auth request');
 };
 try{
  mode='lost';await assert.rejects(activateAuth(invite,'valid password 123'));assert.ok(user);mode='ok';assert.equal(await activateAuth(invite,'valid password 123'),invite.userId);assert.equal(creates,1);assert.equal(signins,1);
  mode='wrong-password';await assert.rejects(activateAuth(invite,'different password'),/earlier activation/);
  mode='ok';assert.equal(await activateAuth({...invite,existing:true},undefined,session(user).access_token),invite.userId);
  mode='wrong-email';await assert.rejects(activateAuth(invite,undefined,session(user).access_token),/email address/);
  mode='ok';user.app_metadata={};await assert.rejects(activateAuth(invite,'valid password 123'),/existing account/);
 }finally{globalThis.fetch=previous;for(const [key,value] of Object.entries(env)){if(value===undefined)delete process.env[key];else process.env[key]=value;}}
});
