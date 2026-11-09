export function GET() {
 return Response.json({status:'ok',service:'alphatrace-web',scope:'web-process-only',database:'not_checked',workers:'not_checked'},{headers:{'Cache-Control':'no-store'}});
}
