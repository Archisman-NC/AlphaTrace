'use client';
import type {ImportReview} from '@/lib/provider-review-contract';
export default function ImportEvidenceContext({item}:{item:ImportReview}){
 return <section className="panel import-evidence-context" aria-labelledby="import-context-title">
  <div className="panel-heading"><h2 id="import-context-title">IndianAPI observation · {item.symbol}</h2><a href={`/imports?review=${item.id}#import-review`} target="_blank" rel="noreferrer">Return to exact import ↗</a></div>
  <div className="import-panel-body"><p>Observed {new Date(item.observed_at).toLocaleString('en-IN',{dateStyle:'medium',timeStyle:'short',timeZone:'Asia/Kolkata'})} IST. Provider periods: {item.evaluation.periods.join(' → ')}.</p><p>{item.evaluation.note}</p>
   {item.handoff&&<p>Linked evidence {item.handoff.id.slice(0,8)} · {item.handoff.state.replace('_',' ')} · period {item.handoff.periodEnd}.</p>}
   {!item.isCurrent&&<p className="monitor-failure">A newer provider change exists. This observation is retained for history; use the latest change before publication.</p>}
   {item.handoff&&item.handoff.periodEnd!==item.evaluation.periods.at(-1)&&<p className="monitor-failure">The evidence template still covers {item.handoff.periodEnd}. Attach the documents and update all dates for {item.evaluation.periods.at(-1)} before approval.</p>}
   <p>Use the values below to investigate. Enter verified reported values in the evidence workbench; provider profit definitions may differ.</p>
  </div>
  <div className="import-table-scroll"><table className="import-table"><caption className="sr-only">Detected provider changes, separate from official evidence</caption><thead><tr><th>Quarter</th><th>Provider metric</th><th>Before</th><th>After</th></tr></thead><tbody>{item.diff.changes.map(c=><tr key={c.period+c.metric}><th>{c.period}</th><td>{c.metric}</td><td>{item.diff.baseline?'No baseline':c.before??'Missing'}</td><td>{c.after??'Missing'}</td></tr>)}</tbody></table></div>
  <div className="import-panel-body"><ul>{item.evaluation.rows.map(r=><li key={r.label}><strong>{r.label}:</strong> {r.definition}</li>)}</ul><p className="fine-print">Evidence approval and publication remain separate from import triage. Published research reaches following users through the existing review inbox.</p></div>
 </section>;
}
