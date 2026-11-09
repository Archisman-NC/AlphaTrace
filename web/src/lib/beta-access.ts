import {asUser,AccountClosedError,AccessRevokedError} from './db';
import {invitedUser} from './invites';
export async function betaAccess(owner:string,allowClosed=false):Promise<boolean>{
 try{return await asUser(owner,async c=>{
  const status=(await c.query('SELECT alphatrace.beta_access_status() AS status')).rows[0].status;
  if(status==='legacy')return invitedUser(owner,process.env.ALPHATRACE_INVITED_USER_IDS);
  return status==='active'||allowClosed&&status==='closed';
 },{accountControl:true,statementTimeoutMs:5000});}catch(e){if(e instanceof AccountClosedError||e instanceof AccessRevokedError)return false;throw e;}
}
export async function betaRecipients(operator:string){return asUser(operator,async c=>{
 const status=(await c.query('SELECT alphatrace.beta_access_status() AS status')).rows[0].status;
 if(status==='legacy')return [...new Set((process.env.ALPHATRACE_INVITED_USER_IDS??'').split(',').map(s=>s.trim()))].filter(id=>invitedUser(id,process.env.ALPHATRACE_INVITED_USER_IDS));
 return (await c.query('SELECT alphatrace.beta_active_users() AS id')).rows.map(r=>r.id as string);
});}
