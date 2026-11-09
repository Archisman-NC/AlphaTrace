import {asUser} from './db';
import {exportJournalWithClient} from './journal';
import {JournalLimitError} from './journal-format';
const queries={
 onboarding:'SELECT version,step,status,topics,updated_at FROM alphatrace.user_onboarding WHERE owner_id=$1',
 emailPreferences:'SELECT version,enabled,reminders,enabled_since,suppressed,updated_at FROM alphatrace.email_preferences WHERE owner_id=$1',
 emailHistory:'SELECT id,digest_day,recipient,state,items,attempts,message,created_at,accepted_at,delivered_at,bounced_at,complained_at FROM alphatrace.email_deliveries WHERE owner_id=$1 ORDER BY created_at,id',
 issueReports:'SELECT id,publication_id,metric,category,note,context,status,version,outcome,response,correction_id,created_at,updated_at FROM alphatrace.research_issues WHERE owner_id=$1 ORDER BY created_at,id',
 issueEvents:'SELECT e.issue_id,e.version,e.action,e.note,e.outcome,e.correction_id,e.created_at FROM alphatrace.research_issue_events e JOIN alphatrace.research_issues i ON i.id=e.issue_id WHERE i.owner_id=$1 ORDER BY e.issue_id,e.version',
 issuePublications:`SELECT id,symbol,version,payload,published_at,withdrawn_at,withdrawal_reason FROM alphatrace.research_updates WHERE id IN (SELECT publication_id FROM alphatrace.research_issues WHERE owner_id=$1 UNION SELECT correction_id FROM alphatrace.research_issues WHERE owner_id=$1 UNION SELECT e.correction_id FROM alphatrace.research_issue_events e JOIN alphatrace.research_issues i ON i.id=e.issue_id WHERE i.owner_id=$1) ORDER BY symbol,version`,
};
export async function exportAccount(owner:string){return asUser(owner,async c=>{
 const journal=await exportJournalWithClient(c,owner),data:Record<string,unknown>={};let bytes=Buffer.byteLength(JSON.stringify(journal));
 for(const [name,sql] of Object.entries(queries)){
  const stats=(await c.query(`SELECT count(*)::int AS count,coalesce(sum(octet_length(row_to_json(t)::text)),0)::text AS bytes FROM (${sql} LIMIT 5001) t`,[owner])).rows[0];bytes+=Number(stats.bytes);
  if(stats.count>5000||bytes>10*1024*1024)throw new JournalLimitError('Your data exceeds the download limit. No partial export was created. Contact the beta operator for a full export.');
  data[name]=(await c.query(sql,[owner])).rows;
 }
 return JSON.parse(JSON.stringify({schemaVersion:1,exportedAt:journal.exportedAt,complete:true,scope:'Saved private account data. Excludes passwords, sessions, unsubscribe capabilities, provider logs and shared operator work. Not a restore file.',journal,...data}));
},{readOnlySnapshot:true});}
