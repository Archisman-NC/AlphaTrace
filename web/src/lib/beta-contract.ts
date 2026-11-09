export type BetaPerson={id:string;email:string|null;version:number;state:'invited'|'active'|'revoked'|'expired'|'closed';createdAt:string;expiresAt:string|null;activatedAt:string|null;isOperator:boolean;setupCompleted:boolean;firstResponseSaved:boolean};
export type BetaPeople={initialized:boolean;limit:number;used:number;checkedAt:string;people:BetaPerson[]};
export type BetaInvitation={email:string;expiresAt:string;existing:boolean};
export class BetaInputError extends Error{}
export class BetaConflictError extends Error{}
export class BetaForbiddenError extends Error{}
export class BetaInvitationError extends Error{}
const uuid=/^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
export function betaEmail(value:unknown){if(typeof value!=='string'||value.length>254||!/^\S+@[^\s@]+\.[^\s@]+$/.test(value.trim())||/[\x00-\x20\x7f]/.test(value.trim()))throw new BetaInputError('Enter a valid email address.');return value.trim().toLowerCase();}
export function betaToken(value:unknown){if(typeof value!=='string'||!/^[A-Za-z0-9_-]{43}$/.test(value))throw new BetaInvitationError('This invitation is unavailable. Ask the operator for a new link.');return value;}
export type BetaCommand={action:'invite';id:string;email:string;token:string}|{action:'reissue'|'revoke';id:string;version:number;token?:string};
export function betaCommand(input:unknown):BetaCommand{
 const b=input as Record<string,unknown>;if(!b||!uuid.test(String(b.id)))throw new BetaInputError('Invalid invitation identifier.');
 if(b.action==='invite')return {action:'invite',id:String(b.id),email:betaEmail(b.email),token:betaToken(b.token)};
 if(!['reissue','revoke'].includes(String(b.action))||!Number.isSafeInteger(b.version)||(b.version as number)<1)throw new BetaInputError('Refresh People before changing access.');
 if(b.action==='revoke'&&b.confirmed!==true)throw new BetaInputError('Confirm that you want to revoke access.');
 return {action:b.action as 'reissue'|'revoke',id:String(b.id),version:b.version as number,...(b.action==='reissue'?{token:betaToken(b.token)}:{})};
}
export function newInvitationToken(){const bytes=crypto.getRandomValues(new Uint8Array(32));return btoa(String.fromCharCode(...bytes)).replaceAll('+','-').replaceAll('/','_').replace(/=+$/,'');}
