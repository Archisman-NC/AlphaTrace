import test from 'node:test';
import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { validateThesis,emptyThesis } from '../src/lib/thesis-input';
const input=()=>({symbol:'INFY',version:0,commandId:randomUUID(),payload:{...emptyThesis(),thesis:'My own research'},reason:'Initial research',confirmed:false});
test('notes work without assumptions, coverage or an AI provider',()=>{assert.equal(validateThesis(input()).payload.thesis,'My own research');});
test('assumptions require explicit confirmation, preserve blanks and stable identity',()=>{
 const x=input(),id=randomUUID();x.payload.assumptions=[{id,statement:'My assumption',metric:'',baseline:'',direction:'unspecified',horizon:'',reviewCondition:''}];
 assert.throws(()=>validateThesis(x),/Confirm/);x.confirmed=true;
 const saved=validateThesis(x);assert.equal(saved.payload.assumptions[0].id,id);assert.equal(saved.payload.assumptions[0].baseline,'');
 x.payload.assumptions.push({...x.payload.assumptions[0]});assert.throws(()=>validateThesis(x),/unique/);
});
test('bounded notes, known company, exact revisions and valid evidence identifiers',()=>{
 const x=input();x.payload.thesis='';assert.throws(()=>validateThesis(x),/Write your reasoning/);
 x.payload.thesis='a'.repeat(6001);assert.throws(()=>validateThesis(x),/characters/);
 x.payload.thesis='Research';x.symbol='UNKNOWN';assert.throws(()=>validateThesis(x),/universe/);
 x.symbol='INFY';x.version=0.5;assert.throws(()=>validateThesis(x),/revision/);
 x.version=0;x.payload.evidenceIds=['not-a-publication-id'];assert.throws(()=>validateThesis(x),/identifier/);
});
test('caller-supplied ownership is never part of the validated command',()=>{const x={...input(),ownerId:randomUUID()};assert.equal('ownerId' in validateThesis(x),false);});
