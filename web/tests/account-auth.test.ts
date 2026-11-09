import {test} from 'node:test';
import assert from 'node:assert/strict';
import {randomUUID} from 'node:crypto';
import {reauthenticate,removeAccountLogin} from '../src/lib/account-auth';
test('account Auth verifies identity and password, removes only the requested login, and handles missing users idempotently',async()=>{
 const originalFetch=globalThis.fetch,env={SUPABASE_URL:process.env.SUPABASE_URL,SUPABASE_PUBLISHABLE_KEY:process.env.SUPABASE_PUBLISHABLE_KEY,SUPABASE_SECRET_KEY:process.env.SUPABASE_SECRET_KEY};
 Object.assign(process.env,{SUPABASE_URL:'https://synthetic.supabase.co',SUPABASE_PUBLISHABLE_KEY:'sb_publishable_synthetic',SUPABASE_SECRET_KEY:'sb_secret_synthetic'});
 const id=randomUUID(),other=randomUUID(),user={id,email:'synthetic@example.test',aud:'authenticated',role:'authenticated',created_at:new Date().toISOString()},token='eyJhbGciOiJIUzI1NiJ9.'+Buffer.from(JSON.stringify({sub:id,exp:Math.floor(Date.now()/1000)+3600})).toString('base64url')+'.signature';let mode='ok',deleteCalls=0;
 globalThis.fetch=async(input,init)=>{const url=String(input),body=init?.body?JSON.parse(String(init.body)):null;
  if(url.includes('/admin/users/')){deleteCalls++;assert.ok(url.endsWith(id));assert.equal(body?.should_soft_delete,false);return mode==='missing'?Response.json({code:'user_not_found',msg:'User not found'},{status:404,headers:{'x-supabase-api-version':'2024-01-01'}}):mode==='outage'?Response.json({msg:'Synthetic outage'},{status:503}):Response.json({user});}
  if(url.includes('/token')){assert.equal(body.email,user.email);if(body.password!=='correct password')return Response.json({code:'invalid_credentials',msg:'Invalid credentials'},{status:400});return Response.json({access_token:token,refresh_token:'synthetic-refresh',expires_in:3600,token_type:'bearer',user:mode==='mismatch'?{...user,id:other}:user});}
  if(url.includes('/logout'))return new Response(null,{status:204});
  if(url.includes('/user'))return Response.json(user);
  throw Error('Unexpected synthetic Auth request');
 };
 try{
  await assert.rejects(reauthenticate(other,token,'correct password'),/session expired/);
  await assert.rejects(reauthenticate(id,token,'wrong password'),/could not be verified/);
  mode='mismatch';await assert.rejects(reauthenticate(id,token,'correct password'),/could not be verified/);
  mode='ok';const client=await reauthenticate(id,token,'correct password');await client.auth.signOut({scope:'local'});
  await removeAccountLogin(id);mode='missing';await removeAccountLogin(id);mode='outage';await assert.rejects(removeAccountLogin(id),/pending retry/);assert.equal(deleteCalls,3);
 }finally{globalThis.fetch=originalFetch;for(const [key,value] of Object.entries(env)){if(value===undefined)delete process.env[key];else process.env[key]=value;}}
});
