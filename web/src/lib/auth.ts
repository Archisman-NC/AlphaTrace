import 'server-only';
import { createClient } from '@supabase/supabase-js';
import { invitedUser,publicAuthConfig } from './invites';
import { localAccessAllowed, validDevToken, UUID } from './access';
export async function requestIdentity(request:Request):Promise<string|null> {
  const token=request.headers.get('authorization')?.match(/^Bearer (.+)$/)?.[1];
  if (!token) return null;
  if (localAccessAllowed(process.env,request.headers.get('host')) && validDevToken(token,process.env.ALPHATRACE_DEV_TOKEN)) {
    const id=process.env.ALPHATRACE_DEV_USER_ID;
    return id && UUID.test(id) ? id : null;
  }
  const config=publicAuthConfig(process.env);
  if (!config) return null;
  const {url,key}=config;
  try {
    const auth=createClient(url,key,{auth:{persistSession:false,autoRefreshToken:false}});
    const {data,error}=await auth.auth.getUser(token);
    return !error && data.user && data.user.email_confirmed_at && invitedUser(data.user.id,process.env.ALPHATRACE_INVITED_USER_IDS) ? data.user.id : null;
  } catch { return null; }
}
