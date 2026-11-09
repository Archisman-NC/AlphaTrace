import test from 'node:test';
import assert from 'node:assert/strict';
import { providerRunOutcome,type ProviderRun } from '../src/lib/provider-contract';
const run={state:'success'} as ProviderRun;
test('only an explicitly evaluated unchanged run claims no tracked changes',()=>{
 assert.equal(providerRunOutcome({...run,detection_result:'unchanged'}),'No tracked changes');
 assert.match(providerRunOutcome(run),/not recorded/);
 assert.match(providerRunOutcome({...run,detection_result:'stale'}),/Older observation/);
 assert.match(providerRunOutcome({...run,detection_result:'baseline'}),/Baseline review/);
 assert.match(providerRunOutcome({...run,detection_result:'changed'}),/sent to review/);
});
test('unfinished and failed requests never present a financial comparison as successful',()=>{
 assert.equal(providerRunOutcome({...run,state:'running'}),'Fetching data');
 assert.match(providerRunOutcome({...run,state:'failed',detection_result:'unchanged'}),/Retrieval failed/);
});
