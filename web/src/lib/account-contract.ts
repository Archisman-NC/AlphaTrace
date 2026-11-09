export class AccountInputError extends Error {}
export class AccountConflictError extends Error {}
export type AccountState={closed:boolean;completed:boolean;blocked:boolean};
export function deletionInput(input:unknown){
 const v=input as Record<string,unknown>;
 if(!v||v.confirmation!=='DELETE MY ACCOUNT'||v.acknowledged!==true)throw new AccountInputError('Type DELETE MY ACCOUNT and confirm that you understand what will be removed.');
 if(typeof v.password!=='string'||!v.password||v.password.length>128)throw new AccountInputError('Enter your current password.');
 return {password:v.password};
}
export function passwordInput(input:unknown){
 const v=input as Record<string,unknown>;
 if(!v||typeof v.currentPassword!=='string'||!v.currentPassword||v.currentPassword.length>128)throw new AccountInputError('Enter your current password.');
 if(typeof v.password!=='string'||v.password.length<12||v.password.length>128)throw new AccountInputError('Use a new password between 12 and 128 characters.');
 if(v.password!==v.confirmPassword)throw new AccountInputError('The new passwords do not match.');
 if(v.password===v.currentPassword)throw new AccountInputError('Choose a different password from your current one.');
 return {currentPassword:v.currentPassword,password:v.password};
}
