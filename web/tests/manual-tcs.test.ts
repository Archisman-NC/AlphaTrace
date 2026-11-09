import test from 'node:test';
import assert from 'node:assert/strict';
import { manualDownloadTime,validateManualFiles,tcsImportFiles,manualRequestLimit } from '../src/lib/manual-tcs-contract';
import { boundedManualForm } from '../src/lib/manual-tcs';
function form(){const data=new FormData();data.set('originalsConfirmed','true');for(const spec of tcsImportFiles){data.set(spec.key,new File(['fixture'],spec.key+spec.extension));data.set(spec.key+'Time',new Date().toISOString());}return data;}
test('manual originals require every file, explicit provenance and actual timezone-aware recent timestamps',()=>{
 const data=form();assert.equal(validateManualFiles(data).length,3);
 for(const mutation of [(f:FormData)=>f.delete('q2'),(f:FormData)=>f.set('q1',new File(['html'],'q1.html')),(f:FormData)=>f.set('originalsConfirmed','false'),(f:FormData)=>f.append('q1',new File(['x'],'q1.pdf')),(f:FormData)=>f.set('publish','true')]){const bad=form();mutation(bad);assert.throws(()=>validateManualFiles(bad));}
 const now=Date.parse('2026-11-08T10:00:00Z');assert.equal(manualDownloadTime('2026-11-08T15:30+05:30',now),'2026-11-08T10:00:00.000Z');
 for(const time of ['2026-11-08T10:00','2026-11-08T10:06Z','2026-10-30T10:00Z','not a date'])assert.throws(()=>manualDownloadTime(time,now));
 assert.throws(()=>manualDownloadTime('2026-02-30T10:00Z',Date.parse('2026-03-02T10:00Z')),/calendar date/);
});
test('multipart imports reject wrong content types, declared oversize and malformed bodies',async()=>{
 await assert.rejects(boundedManualForm(new Request('http://localhost',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'})),/three original files/);
 await assert.rejects(boundedManualForm(new Request('http://localhost',{method:'POST',headers:{'Content-Type':'multipart/form-data; boundary=x','Content-Length':String(manualRequestLimit+1)},body:'x'})),/exceeds/);
 await assert.rejects(boundedManualForm(new Request('http://localhost',{method:'POST',headers:{'Content-Type':'multipart/form-data; boundary=x'},body:'broken'})),/Could not read/);
 const parsed=await boundedManualForm(new Request('http://localhost',{method:'POST',body:form()}));assert.equal(validateManualFiles(parsed).length,3);
});
