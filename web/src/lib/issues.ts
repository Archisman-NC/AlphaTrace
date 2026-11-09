import type { PoolClient } from 'pg';
import { asUser } from './db';
import { issueQuery,validateIssue,IssueInputError,IssueConflictError,IssueForbiddenError,type IssueQuery,type IssueView,type IssueDetail } from './issue-contract';
const serial=<T>(v:T):T=>JSON.parse(JSON.stringify(v));
const columns=`i.id,i.publication_id AS "publicationId",i.metric,i.category,i.note,i.context,i.status,i.version,i.outcome,i.response,i.correction_id AS "correctionId",
 to_char(i.created_at AT TIME ZONE 'UTC','YYYY-MM-DD"T"HH24:MI:SS.US"Z"') AS "createdAt",i.updated_at AS "updatedAt",u.withdrawn_at AS "withdrawnAt",fix.withdrawn_at AS "correctionWithdrawnAt"`;
const joins='FROM alphatrace.research_issues i JOIN alphatrace.research_updates u ON u.id=i.publication_id LEFT JOIN alphatrace.research_updates fix ON fix.id=i.correction_id';
async function operator(c:PoolClient,id:string){return !!(await c.query('SELECT 1 FROM alphatrace.research_operators WHERE user_id=$1',[id])).rowCount;}
async function detail(c:PoolClient,id:string,isOperator:boolean):Promise<IssueDetail>{
 const report=(await c.query(`SELECT ${columns} ${joins} WHERE i.id=$1`,[id])).rows[0];
 if(!report)throw new IssueInputError('This report is unavailable to your account.');
 const events=(await c.query('SELECT version,action,note,outcome,correction_id AS "correctionId",created_at AS "createdAt" FROM alphatrace.research_issue_events WHERE issue_id=$1 ORDER BY version DESC LIMIT 50',[id])).rows;
 const corrections=isOperator?(await c.query(`SELECT id,version,payload->>'title' AS title FROM alphatrace.research_updates WHERE symbol=$1 AND version>$2 AND withdrawn_at IS NULL ORDER BY version DESC LIMIT 25`,[report.context.symbol,report.context.publicationVersion])).rows:[];
 return serial({report,events,corrections});
}
export async function getIssues(owner:string,input:IssueQuery={}):Promise<IssueView>{
 const q=issueQuery(input);
 return asUser(owner,async c=>{
  const isOperator=await operator(c,owner);
  if(q.scope==='queue'&&!isOperator)throw new IssueForbiddenError('The report queue requires research operator access.');
  const rows=(await c.query(`SELECT ${columns} ${joins} WHERE ($1::boolean OR i.owner_id=$2)
   AND ($3='all' OR ($3='active' AND i.status<>'resolved') OR i.status=$3)
   AND ($4::timestamptz IS NULL OR (i.created_at,i.id)<($4::timestamptz,$5::uuid))
   ORDER BY i.created_at DESC,i.id DESC LIMIT 26`,[q.scope==='queue',owner,q.status,q.cursor?.[0]??null,q.cursor?.[1]??null])).rows;
  const counts=(await c.query(`SELECT count(*) FILTER(WHERE status='open')::int AS open,count(*) FILTER(WHERE status='in_review')::int AS in_review,count(*) FILTER(WHERE status='resolved')::int AS resolved
   FROM alphatrace.research_issues WHERE ($1::boolean OR owner_id=$2)`,[q.scope==='queue',owner])).rows[0];
  const items=rows.slice(0,25),last=items.at(-1);
  return serial({items,nextCursor:rows.length>25?`${last.createdAt}|${last.id}`:null,isOperator,counts,detail:q.report?await detail(c,q.report,isOperator):null});
 },{readOnlySnapshot:true});
}
export async function saveIssue(owner:string,input:unknown){
 const command=validateIssue(input);
 return asUser(owner,async c=>{
  const isOperator=await operator(c,owner);
  if(command.action!=='submit'&&!isOperator)throw new IssueForbiddenError('Only research operators can update report status.');
  await c.query('SELECT pg_advisory_xact_lock(hashtext($1))',['issue-command:'+owner]);
  const retry=(await c.query('SELECT issue_id,command=$3::jsonb AS matches FROM alphatrace.research_issue_events WHERE actor_id=$1 AND command_id=$2',[owner,command.commandId,JSON.stringify(command)])).rows[0];
  if(retry){if(!retry.matches)throw new IssueConflictError('This save identifier was already used for another report action. Reload before continuing.');return {detail:await detail(c,retry.issue_id,isOperator),duplicate:false};}
  if(command.action==='submit'){
   const publication=(await c.query('SELECT payload FROM alphatrace.research_updates WHERE id=$1',[command.publicationId])).rows[0];
   if(!publication)throw new IssueInputError('Choose a published research version. Drafts cannot be reported here.');
   if(command.metric&&!publication.payload.comparisons.some((v:{metric:string})=>v.metric===command.metric))throw new IssueInputError('This metric does not belong to the selected publication.');
   const existing=(await c.query("SELECT id FROM alphatrace.research_issues WHERE owner_id=$1 AND publication_id=$2 AND metric IS NOT DISTINCT FROM $3 AND category=$4 AND status<>'resolved'",[owner,command.publicationId,command.metric,command.category])).rows[0];
   if(existing)return {detail:await detail(c,existing.id,isOperator),duplicate:true};
   const count=(await c.query("SELECT count(*)::int AS n FROM alphatrace.research_issues WHERE owner_id=$1 AND created_at>now()-interval '24 hours'",[owner])).rows[0].n;
   if(count>=20)throw new IssueInputError('You have submitted 20 reports in the last 24 hours. Please wait before sending another.');
   await c.query("INSERT INTO alphatrace.workspaces(owner_id,name) VALUES($1,'My research') ON CONFLICT DO NOTHING",[owner]);
   const row=(await c.query(`INSERT INTO alphatrace.research_issues(owner_id,publication_id,metric,category,note,last_command_id,last_command,last_note) VALUES($1,$2,$3,$4,$5,$6,$7,$5) RETURNING id`,[owner,command.publicationId,command.metric,command.category,command.note,command.commandId,JSON.stringify(command)])).rows[0];
   return {detail:await detail(c,row.id,isOperator),duplicate:false};
  }
  const current=(await c.query('SELECT * FROM alphatrace.research_issues WHERE id=$1 FOR UPDATE',[command.id])).rows[0];
  if(!current)throw new IssueInputError('This report is unavailable.');
  if(current.version!==command.version)throw new IssueConflictError('This report changed in another tab. Reload it before saving your response.');
  if(command.action==='in_review'&&current.status!=='open'||command.action==='resolve'&&current.status==='resolved'||command.action==='reopen'&&current.status!=='resolved')throw new IssueConflictError('This action no longer matches the report status. Reload the report.');
  if(command.action==='reopen'){
   const duplicate=(await c.query("SELECT id FROM alphatrace.research_issues WHERE owner_id=$1 AND publication_id=$2 AND metric IS NOT DISTINCT FROM $3 AND category=$4 AND status<>'resolved'",[current.owner_id,current.publication_id,current.metric,current.category])).rows[0];
   if(duplicate)throw new IssueConflictError('Another active report already covers this evidence. Review that report instead.');
  }
  if(command.action==='resolve'){
   if(command.outcome==='corrected'){
    // Serialize with withdrawal of the referenced correction.
    await c.query('SELECT pg_advisory_xact_lock_shared(hashtext($1))',['publication:'+command.correctionId]);
    const correction=(await c.query('SELECT symbol,version,withdrawn_at FROM alphatrace.research_updates WHERE id=$1',[command.correctionId])).rows[0];
    if(!correction||correction.symbol!==current.context.symbol||correction.version<=current.context.publicationVersion||correction.withdrawn_at)throw new IssueInputError('Select a newer, non-withdrawn publication for the same company.');
   }
   if(command.outcome==='withdrawn'&&!(await c.query('SELECT withdrawn_at FROM alphatrace.research_updates WHERE id=$1',[current.publication_id])).rows[0]?.withdrawn_at)throw new IssueInputError('Withdraw the unreliable publication in the research desk before using this resolution.');
  }
  const status=command.action==='resolve'?'resolved':command.action==='reopen'?'open':'in_review';
  await c.query(`UPDATE alphatrace.research_issues SET status=$2,version=version+1,outcome=$3,response=$4,correction_id=$5,last_command_id=$6,last_command=$7,last_note=$8 WHERE id=$1`,[command.id,status,command.outcome,command.action==='resolve'?command.note:null,command.correctionId,command.commandId,JSON.stringify(command),command.note]);
  return {detail:await detail(c,command.id,isOperator),duplicate:false};
 });
}
