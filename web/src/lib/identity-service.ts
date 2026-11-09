import {betaAccess} from './beta-access';
import { asUser,AccountClosedError,AccessRevokedError } from './db';
import { ServiceUnavailableError } from './service-errors';
import { createClient } from '@supabase/supabase-js';
import { publicAuthConfig } from './invites';
import { localAccessAllowed, validDevToken, UUID } from './access';
export async function requestIdentity(request:Request,options:{allowClosed?:boolean}={}):Promise<string|null> {
  const token=request.headers.get('authorization')?.match(/^Bearer (.+)$/)?.[1];
  if (!token) return null;
  if (localAccessAllowed(process.env,request.headers.get('host')) && validDevToken(token,process.env.ALPHATRACE_DEV_TOKEN)) {
    const id=process.env.ALPHATRACE_DEV_USER_ID;
    return id && UUID.test(id) ? id : null;
  }
  if(!/^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$/.test(token))return null;
  const config=publicAuthConfig(process.env);
  if (!config) throw new ServiceUnavailableError();
  const {url,key}=config;
  try {
    const auth=createClient(url,key,{auth:{persistSession:false,autoRefreshToken:false},global:{fetch:(url,init)=>fetch(url,{...init,signal:AbortSignal.timeout(10000)})}});
    const {data,error}=await auth.auth.getUser(token);
    if(error&&(!error.status||error.status>=500||error.status===429))throw new ServiceUnavailableError();
    if(error||!data.user?.email_confirmed_at||!await betaAccess(data.user.id,!!options.allowClosed))return null;
    if(!options.allowClosed)await asUser(data.user.id,async()=>true,{statementTimeoutMs:5000});
    return data.user.id;
  } catch(error) { if(error instanceof AccountClosedError||error instanceof AccessRevokedError)return null;throw new ServiceUnavailableError(); }
}
