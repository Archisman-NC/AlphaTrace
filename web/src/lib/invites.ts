import { UUID } from './access';
/** Server-owned pilot allowlist. Empty, malformed or oversized lists fail closed. */
export function invitedUser(id:string,configuration:string|undefined):boolean {
 const ids=(configuration??'').split(',').map(s=>s.trim().toLowerCase()).filter(Boolean);
 return ids.length>0&&ids.length<=5&&ids.every(s=>UUID.test(s))&&UUID.test(id)&&ids.includes(id.toLowerCase());
}
export function publicAuthConfig(env:Record<string,string|undefined>):{url:string;key:string}|null {
 const url=env.SUPABASE_URL,key=env.SUPABASE_PUBLISHABLE_KEY;
 if(!url||!key)return null;
 try{
  const parsed=new URL(url);
  if(parsed.protocol!=='https:'||parsed.username||parsed.password||parsed.search||parsed.hash)return null;
  const publishable=key.startsWith('sb_publishable_');
  // Never send service-role, secret, or arbitrary JWT keys to the browser.
  const anon=!publishable&&JSON.parse(Buffer.from(key.split('.')[1]??'','base64url').toString()).role==='anon';
  return publishable||anon?{url,key}:null;
 }catch{return null;}
}
