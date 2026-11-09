'use client';
export default function RecoveryScreen({retry}:{retry:()=>void}){
 return <main className="recovery-screen"><a href="/workspace" className="recovery-brand">αT · AlphaTrace</a><section><span className="eyebrow">LET’S GET YOU BACK TO YOUR RESEARCH</span><h1>This page couldn’t be displayed.</h1><p>Try loading it again. If you had just submitted a change, check its saved status before repeating it.</p><p className="fine-print">Unsaved edits may need to be entered again. This error does not confirm whether a save completed.</p><div className="recovery-actions"><button className="primary" onClick={retry}>Try loading again</button><a href="/workspace">Return to workspace →</a></div></section></main>;
}
