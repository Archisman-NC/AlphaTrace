import { PortfolioInputError } from './portfolio-input';
export type EmailPreferences={version:number;enabled:boolean;reminders:boolean;enabledSince:string|null;suppressed:boolean};
export type EmailDelivery={id:string;day:string;state:string;message:string;createdAt:string;acceptedAt:string|null;deliveredAt:string|null;bouncedAt:string|null;complainedAt:string|null};
export type EmailView={preferences:EmailPreferences;deliveries:EmailDelivery[];available:boolean};
export type EmailSettingsResult={email?:EmailView;error?:string;status?:number};
export type DigestItem={key:string;updateId:string;symbol:string;title:string;periodEnd:string;kind:'publication'|'reminder';reviewVersion:number|null};
export type EmailPayload={from:string;to:string[];subject:string;text:string;html:string;headers:Record<string,string>};
export type EmailConfig={from:string;origin:string;apiKey:string};
export function validateEmailPreferences(input:unknown){
 if(!input||typeof input!=='object')throw new PortfolioInputError('Invalid email preferences.');
 const v=input as Record<string,unknown>;
 if(!Number.isSafeInteger(v.version)||(v.version as number)<0||typeof v.enabled!=='boolean'||typeof v.reminders!=='boolean')throw new PortfolioInputError('Reload your email preferences before saving.');
 if(v.enabled&&v.consent!==true)throw new PortfolioInputError('Confirm that you want personal research emails.');
 return {version:v.version as number,enabled:v.enabled,reminders:v.reminders};
}
export function emailConfig(env:Record<string,string|undefined>):EmailConfig{
 const from=env.RESEND_FROM_EMAIL?.trim(),apiKey=env.RESEND_API_KEY?.trim();
 if(!from||!apiKey?.startsWith('re_')||!/^\S+@[^\s<>]+\.[^\s<>]+$/.test(from)||/[\r\n]/.test(from))throw Error('Email sender is not configured.');
 const url=new URL(env.ALPHATRACE_APP_URL??'');
 if(url.protocol!=='https:'||url.username||url.password||url.pathname!=='/'||url.search||url.hash)throw Error('Use the production HTTPS origin for email links.');
 return {from:`AlphaTrace <${from}>`,origin:url.origin,apiKey};
}
export function emailsAvailable(env:Record<string,string|undefined>):boolean{
 if(env.ALPHATRACE_EMAILS_ENABLED!=='true')return false;
 // Resend's shared test sender cannot deliver to other beta users.
 if(env.RESEND_FROM_EMAIL?.trim().toLowerCase().endsWith('@resend.dev'))return false;
 try{emailConfig(env);return !!env.RESEND_WEBHOOK_SECRET?.startsWith('whsec_')&&!!env.SUPABASE_SECRET_KEY;}catch{return false;}
}
export const htmlEscape=(text:string)=>text.replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]!));
export function digestPayload(config:EmailConfig,recipient:string,items:DigestItem[],token:string):EmailPayload{
 if(!items.length||items.length>50)throw Error('A digest must have between 1 and 50 items.');
 const unsubscribe=`${config.origin}/email/unsubscribe?token=${token}`,oneClick=`${config.origin}/api/email/unsubscribe?token=${token}`;
 const link=(i:DigestItem)=>`${config.origin}/inbox?filter=all&update=${encodeURIComponent(i.updateId)}`;
 const label=(i:DigestItem)=>i.kind==='reminder'?'Review reminder':'Published update';
 const subject=`Your AlphaTrace brief · ${items.length} ${items.length===1?'update':'updates'}`;
 const text=['YOUR ALPHATRACE BRIEF','',...items.map(i=>`${label(i)} · ${i.symbol}\n${i.title}\nQuarter ended ${i.periodEnd}\n${link(i)}\n`),'Read the evidence and choose your own response. Opening an update does not mark it reviewed.','',`Email preferences: ${config.origin}/settings`,`Unsubscribe: ${unsubscribe}`,'You opted in to updates for your tracked companies. No automatic investment conclusions.'].join('\n');
 const html=`<!doctype html><html lang="en"><body style="margin:0;background:#f5f6f8;color:#20242c;font-family:Arial,sans-serif"><main style="max-width:600px;margin:0 auto;padding:36px 24px"><p style="font-size:13px;letter-spacing:2px">ALPHATRACE</p><h1 style="font-size:28px;line-height:1.3">Your research, ready to revisit.</h1><p style="line-height:1.7;color:#505966">New evidence and reminders for the companies you track.</p>${items.map(i=>`<section style="padding:24px 0;border-top:1px solid #d6dbe4"><p style="font-size:12px;color:#505966">${label(i)} · ${htmlEscape(i.symbol)}</p><h2 style="font-size:19px;line-height:1.5">${htmlEscape(i.title)}</h2><p style="font-size:13px">Quarter ended ${htmlEscape(i.periodEnd)}</p><a style="color:#414ca1;font-weight:bold" href="${htmlEscape(link(i))}">Open evidence &amp; review →</a></section>`).join('')}<p style="font-size:13px;line-height:1.7">Choose your own response after reading the evidence. Opening an update does not mark it reviewed.</p><footer style="border-top:1px solid #d6dbe4;padding-top:24px;font-size:12px;line-height:1.8;color:#505966">You opted in to updates for your tracked companies. No automatic investment conclusions.<br><a href="${config.origin}/settings">Email preferences</a> · <a href="${unsubscribe}">Unsubscribe</a></footer></main></body></html>`;
 return {from:config.from,to:[recipient],subject,text,html,headers:{'List-Unsubscribe':`<${oneClick}>`,'List-Unsubscribe-Post':'List-Unsubscribe=One-Click'}};
}
