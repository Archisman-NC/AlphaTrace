import { comparisonResult,type EvidencePoint,type ResearchPayload } from '@/lib/research-contract';
import { metricLabel } from '@/lib/inbox-contract';
import { displayDate,evidenceNumber } from '@/lib/dashboard-view';

function Point({point,payload,label}:{point:EvidencePoint;payload:ResearchPayload;label:string}) {
 const source=payload.sources.find(s=>s.id===point.sourceId)!;
 return <div className="update-point"><span className="update-point-label">{label} · {displayDate(point.periodEnd)}</span><strong>{evidenceNumber(point.value)}</strong><small>{point.unit}{point.scope&&` · ${point.scope}`}</small><a href={source.url+(point.page?`#page=${point.page}`:'')} target="_blank" rel="noreferrer" aria-label={`${label} source: ${source.title}${point.page?`, page ${point.page}`:''}`}>{point.page?`Source · p. ${point.page}`:'Source section'} ↗</a></div>;
}

export default function UpdateEvidence({payload}:{payload:ResearchPayload}) {
 const comparable=payload.comparisons.filter(c=>comparisonResult(c,payload.comparison).available).length;
 return <><div className="evidence-section-heading"><div><span className="eyebrow">01 / THE EVIDENCE</span><h3>What changed this quarter</h3></div><span className="badge">{payload.comparison==='QoQ'?'Quarter on quarter':'Year on year'}</span></div>
  <p className="evidence-summary">{comparable} of {payload.comparisons.length} metrics are comparable.{comparable<payload.comparisons.length?' Read the basis differences before interpreting the remaining figures.':' Changes describe reported figures, not investment outcomes.'}</p>
  <div className="update-metrics">{payload.comparisons.map(c=>{
   const result=comparisonResult(c,payload.comparison),change=result.available?`${!result.change!.startsWith('-')&&result.change!=='0.00'?'+':''}${result.change}${result.changeUnit==='pp'?' pp':'%'}`:null;
   return <section className="update-metric" key={c.metric} aria-label={metricLabel(c.metric)}><div className="update-metric-title"><h4>{metricLabel(c.metric)}</h4>{change?<strong className="metric-delta">{change}</strong>:<span className="badge amber">Not comparable</span>}</div><div className="update-periods"><Point label="Previous" point={c.previous} payload={payload}/><span className="period-arrow" aria-hidden="true">→</span><Point label="Current" point={c.current} payload={payload}/></div>{!result.available&&<p className="metric-caveat">{result.reason}</p>}
    <details className="metric-basis"><summary>Source locations & comparison basis</summary><dl>{(['previous','current'] as const).map(key=>{const p=c[key];return <div key={key}><dt>{key==='previous'?'Previous':'Current'} source location</dt><dd>{p.locator}</dd><dt>Basis</dt><dd>{p.scope&&`${p.scope} · `}{p.basis} · {p.standard} · {p.treatment} · {p.periodType.replace('_',' ')}</dd></div>;})}</dl>{c.note&&<p>{c.note}</p>}</details>
   </section>;
  })}</div>
  <details className="update-limitations"><summary>What these figures don’t establish · {payload.limitations.length} limitations</summary><ul>{payload.limitations.map((text,i)=><li key={i}>{text}</li>)}</ul></details>
 </>;
}
