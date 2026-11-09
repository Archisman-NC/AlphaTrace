export function readLocalEnvironment(directory?:string):Record<string,string>;
export function writeLocalEnvironment(values:Record<string,string>,directory?:string):void;
export function isolatedTestEnvironment(values:Record<string,string|undefined>):Record<string,string>;
