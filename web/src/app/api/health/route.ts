export function GET() {
  return Response.json({status:'ok',service:'alphatrace-web'},{headers:{'Cache-Control':'no-store'}});
}
