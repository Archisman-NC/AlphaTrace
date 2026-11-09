import test from 'node:test';
import assert from 'node:assert/strict';
import { journalFile,journalFormat,type Journal } from '../src/lib/journal-format';
const empty=():Journal=>({schemaVersion:1,exportedAt:'2026-11-03T01:00:00.000Z',scope:'Private journal',complete:true,workspace:null,portfolio:[],theses:[],thesisRevisions:[],reviews:[],reviewEvents:[],publications:[],sourceChecks:[]});
test('journal formats and empty account output are explicit',()=>{
 assert.equal(journalFormat(null),'json');assert.equal(journalFormat('md'),'md');assert.throws(()=>journalFormat('html'));
 const file=journalFile(empty(),'md');assert.equal(file.filename,'alphatrace-journal-2026-11-03.md');assert.ok(file.content.includes('No saved review items.'));
 assert.deepEqual(JSON.parse(journalFile(empty(),'json').content),empty());
});
test('Markdown preserves user wording as inert text and JSON preserves exact values',()=>{
 const j=empty(),wording='<script>alert(1)</script>\n![image](https://example.com/track)\n# injected heading';
 j.workspace={name:wording,portfolio_version:1,created_at:j.exportedAt,updated_at:j.exportedAt};
 j.portfolio=[{symbol:'TCS',quantity:'999999999999.999999',followed:true,updated_at:j.exportedAt}];
 const md=journalFile(j,'md').content;
 assert.ok(!md.includes('<script>'));assert.ok(!md.includes('![image]'));assert.ok(!md.includes('\n# injected'));
 assert.ok(md.includes('999999999999.999999'));assert.equal(JSON.parse(journalFile(j,'json').content).workspace.name,wording);
});
test('oversized downloads fail without returning a partial file',()=>{
 const j=empty();j.scope='x'.repeat(20*1024*1024);assert.throws(()=>journalFile(j,'json'),/No partial file/);
});
