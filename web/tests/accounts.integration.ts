import assert from 'node:assert/strict';
import {randomUUID} from 'node:crypto';
import pg from 'pg';
import {databaseConfig} from '../src/lib/database-config.mjs';
import {asUser,closeDatabase} from '../src/lib/db';
import {accountState,requestClosure,finishClosure,pendingClosures} from '../src/lib/accounts';
import {exportAccount} from '../src/lib/account-export';
import {updatePortfolio} from '../src/lib/portfolios';
import {saveThesis} from '../src/lib/theses';
import {emptyThesis} from '../src/lib/thesis-input';
import {ingestResearch,reviewResearch} from '../src/lib/research';
import {saveIssue} from '../src/lib/issues';
import {syncInbox,reviewInbox} from '../src/lib/inbox';
import {saveEmailPreferences,prepareEmailDigest} from '../src/lib/emails';
import {fixture,check} from './research-fixture';
const alice=randomUUID(),bob=randomUUID(),operator=randomUUID(),empty=randomUUID(),owners=[alice,bob,operator,empty];
const db=new pg.Client(databaseConfig(process.env.MIGRATION_DATABASE_URL));await db.connect();
try{
 await db.query('INSERT INTO alphatrace.research_operators(user_id) VALUES($1)',[operator]);
 for(const owner of [alice,bob]){await updatePortfolio(owner,{version:0,command:{type:'follow',symbol:'TCS',followed:true}});await saveEmailPreferences(owner,{version:0,enabled:true,reminders:true,consent:true},true);await saveThesis(owner,{symbol:'TCS',version:0,commandId:randomUUID(),payload:{...emptyThesis(),thesis:'PRIVATE '+owner},reason:'Initial synthetic research',confirmed:true});}
 const p=fixture('SYNTHETIC account cleanup '+operator),d=await ingestResearch(operator,{draft:p,check:check(p)});await reviewResearch(operator,{action:'approve',id:d.draftId,note:'Checked synthetic account cleanup evidence only',evidenceConfirmed:true,basisConfirmed:true});const published=(await reviewResearch(operator,{action:'publish',id:d.draftId})).id;
 const mail=await prepareEmailDigest(alice,'synthetic@example.test',{from:'updates@example.test',origin:'https://example.test',apiKey:'unused'});assert.ok(mail);
 for(const owner of [alice,bob]){await syncInbox(owner);await reviewInbox(owner,{updateId:published,commandId:randomUUID(),version:0,action:'keep',note:'PRIVATE response '+owner});await saveIssue(owner,{action:'submit',commandId:randomUUID(),publicationId:published,metric:'revenue',category:'incorrect_value',note:'PRIVATE issue '+owner});}
 await asUser(alice,c=>c.query("INSERT INTO alphatrace.user_onboarding(owner_id,topics) VALUES($1,ARRAY['revenue'])",[alice]));
 await db.query("UPDATE alphatrace.email_deliveries SET provider_id=$2 WHERE id=$1",[mail,'provider-'+mail]);await db.query("INSERT INTO alphatrace.email_webhook_events(event_id,provider_id,event_type) VALUES($1,$2,'email.delivered')",['event-'+mail,'provider-'+mail]);
 const exported=await exportAccount(alice);assert.equal(exported.journal.portfolio.length,1);assert.equal(exported.issueReports.length,1);assert.equal(exported.issueEvents.length,1);assert.equal(exported.onboarding.length,1);assert.equal(exported.emailHistory.length,1);assert.equal(exported.issuePublications.length,1);assert.ok(JSON.stringify(exported).includes('PRIVATE '+alice));assert.ok(!JSON.stringify(exported).includes(bob));assert.ok(!JSON.stringify(exported).includes('unsubscribe_hash'));assert.ok(!JSON.stringify(exported).includes('command_id'));
 assert.equal((await exportAccount(operator)).issueReports.length,0,'operators cannot export other users reports');
 assert.equal((await accountState(operator)).blocked,true);await assert.rejects(requestClosure(operator),/Transfer/);assert.equal((await accountState(operator)).closed,false);
 // An open writer finishes before closure; all later writers fail instead of recreating deleted data.
 let release!:()=>void,started!:()=>void;const held=new Promise<void>(r=>release=r),ready=new Promise<void>(r=>started=r);
 const writer=asUser(alice,async c=>{started();await held;await c.query("UPDATE alphatrace.workspaces SET name='Synthetic final write' WHERE owner_id=$1",[alice]);});await ready;const deletion=requestClosure(alice);release();await writer;await deletion;
 assert.equal((await accountState(alice)).closed,true);assert.equal((await accountState(alice)).completed,false);
 for(const table of ['workspaces','portfolio_entries','private_research','private_research_revisions','review_items','review_events','user_onboarding','email_preferences','email_deliveries','research_issues'])assert.equal((await db.query(`SELECT count(*)::int AS n FROM alphatrace.${table} WHERE owner_id=$1`,[alice])).rows[0].n,0,table);
 assert.equal((await db.query('SELECT 1 FROM alphatrace.email_webhook_events WHERE event_id=$1',['event-'+mail])).rowCount,0);
 assert.equal((await db.query('SELECT 1 FROM alphatrace.research_updates WHERE id=$1',[published])).rowCount,1,'shared publication retained');assert.equal((await exportAccount(bob)).issueReports.length,1,'other owner unaffected');
 await assert.rejects(exportAccount(alice),/closed/);await assert.rejects(updatePortfolio(alice,{version:0,command:{type:'follow',symbol:'TCS',followed:true}}),/closed/);
 await assert.rejects(asUser(alice,c=>c.query("INSERT INTO alphatrace.workspaces(owner_id,name) VALUES($1,'Recreated')",[alice]),{accountControl:true}),/closed/);
 assert.equal(await asUser(bob,async c=>(await c.query('SELECT * FROM alphatrace.account_closures WHERE owner_id=$1',[alice])).rowCount),0);
 let attempts=0;await assert.rejects(finishClosure(alice,async id=>{assert.equal(id,alice);attempts++;throw Error('Synthetic auth outage');}),/Synthetic/);assert.equal(attempts,1);assert.equal(await finishClosure(alice,async()=>{attempts++;}),false,'lease prevents immediate duplicate');
 await db.query('UPDATE alphatrace.account_closures SET next_attempt_at=now() WHERE owner_id=$1',[alice]);assert.ok((await pendingClosures(operator)).includes(alice));assert.equal(await finishClosure(alice,async()=>{attempts++;}),true);assert.equal(await finishClosure(alice,async()=>{attempts++;}),true);assert.equal(attempts,2);assert.equal((await accountState(alice)).completed,true);
 await requestClosure(alice);assert.equal((await accountState(alice)).completed,true,'repeat deletion keeps completed receipt');
 await requestClosure(empty);assert.equal((await accountState(empty)).closed,true,'accounts without a workspace can close');
 await closeDatabase();assert.equal((await accountState(alice)).completed,true);
 console.log('PASS: full private export, two-user isolation, immutable shared research, operator handover guard, concurrent writer/deletion serialization, private cascades, old-session recreation denial and durable Auth retry');
}finally{
 await closeDatabase();await db.query('BEGIN');await db.query('DELETE FROM alphatrace.email_webhook_events WHERE provider_id IN(SELECT provider_id FROM alphatrace.email_deliveries WHERE owner_id=ANY($1::uuid[]))',[owners]);await db.query('DELETE FROM alphatrace.workspaces WHERE owner_id=ANY($1::uuid[])',[owners]);await db.query('DELETE FROM alphatrace.account_closures WHERE owner_id=ANY($1::uuid[])',[owners]);await db.query('ALTER TABLE alphatrace.research_drafts DISABLE TRIGGER check_research_review');await db.query('UPDATE alphatrace.research_drafts SET published_update_id=NULL WHERE created_by=$1',[operator]);await db.query('DELETE FROM alphatrace.research_updates WHERE published_by=$1',[operator]);await db.query('DELETE FROM alphatrace.research_drafts WHERE created_by=$1',[operator]);await db.query('ALTER TABLE alphatrace.research_drafts ENABLE TRIGGER check_research_review');await db.query('DELETE FROM alphatrace.research_checks WHERE recorded_by=$1',[operator]);await db.query('DELETE FROM alphatrace.research_operators WHERE user_id=$1',[operator]);await db.query('COMMIT');await db.end();
}
