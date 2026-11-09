import { asUser } from './db';
import type { PoolClient } from 'pg';
import { JournalLimitError,type Journal } from './journal-format';

// Static SQL projections intentionally omit auth identities, internal commands,
// operator drafts and unrelated publications. RLS remains in force as well.
const queries={
 workspace:'SELECT name,portfolio_version,created_at,updated_at FROM alphatrace.workspaces WHERE owner_id=$1',
 portfolio:'SELECT symbol,quantity::text,followed,updated_at FROM alphatrace.portfolio_entries WHERE owner_id=$1 ORDER BY symbol',
 theses:'SELECT symbol,version,updated_at FROM alphatrace.private_research WHERE owner_id=$1 ORDER BY symbol',
 thesisRevisions:'SELECT symbol,version,payload,reason,created_at FROM alphatrace.private_research_revisions WHERE owner_id=$1 ORDER BY symbol,version',
 reviews:'SELECT update_id,snapshot,version,state,revisit_at,created_at,updated_at FROM alphatrace.review_items WHERE owner_id=$1 ORDER BY created_at,update_id',
 reviewEvents:'SELECT update_id,version,action,note,revisit_at,thesis_revision,created_at FROM alphatrace.review_events WHERE owner_id=$1 ORDER BY update_id,version',
 publications:`SELECT u.id,u.symbol,u.version,u.payload,u.published_at,u.withdrawn_at,u.withdrawal_reason,u.supersedes,
  (SELECT n.id FROM alphatrace.research_updates n WHERE n.symbol=u.symbol AND n.version>u.version ORDER BY n.version DESC LIMIT 1) AS newer_update_id
  FROM alphatrace.research_updates u WHERE u.id IN (
   SELECT update_id FROM alphatrace.review_items WHERE owner_id=$1
   UNION SELECT jsonb_array_elements_text(payload->'evidenceIds')::uuid FROM alphatrace.private_research_revisions WHERE owner_id=$1
  ) ORDER BY u.symbol,u.version`,
 sourceChecks:`SELECT DISTINCT ON(c.symbol) c.symbol,c.status,c.checked_at,c.message,c.fingerprints FROM alphatrace.research_checks c
  WHERE c.symbol IN (SELECT symbol FROM alphatrace.private_research WHERE owner_id=$1 UNION
   SELECT u.symbol FROM alphatrace.review_items i JOIN alphatrace.research_updates u ON u.id=i.update_id WHERE i.owner_id=$1)
  ORDER BY c.symbol,c.checked_at DESC`,
};
export async function exportJournal(owner:string):Promise<Journal>{
 return asUser(owner,c=>exportJournalWithClient(c,owner),{readOnlySnapshot:true});
}
export async function exportJournalWithClient(c:PoolClient,owner:string):Promise<Journal>{
  await c.query("SET LOCAL statement_timeout='15s'");
  const exportedAt=(await c.query('SELECT transaction_timestamp() AS time')).rows[0].time.toISOString();
  const data:Record<string,unknown>={};let bytes=0;
  for(const [name,query] of Object.entries(queries)){
   // Bound each category before transferring private payloads to application memory.
   const stats=(await c.query(`SELECT count(*)::int AS count,coalesce(sum(octet_length(row_to_json(t)::text)),0)::text AS bytes FROM (${query} LIMIT 5001) t`,[owner])).rows[0];
   bytes+=Number(stats.bytes);
   if(stats.count>5000||bytes>10*1024*1024)throw new JournalLimitError('This journal exceeds the download limit. No partial file was created. Ask the operator for a full export.');
   const rows=(await c.query(query,[owner])).rows;
   data[name]=name==='workspace'?(rows[0]??null):rows;
  }
  return JSON.parse(JSON.stringify({schemaVersion:1,exportedAt,scope:'Your saved private portfolio, thesis revisions, review history and the published evidence they reference.',complete:true,...data}));
}
