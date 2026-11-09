import { tasks,configure } from '@trigger.dev/sdk';
import type { alphaTraceHealthCheck } from '../src/trigger/health-check';

// The SDK reads the key from the environment; never print it or pass it in argv.
const key=process.env.TRIGGER_DEV_SECRET_KEY?.trim();
if(!key||!key.startsWith('tr_dev_')) {
  console.error('Save a Trigger.dev Development secret key in web/.env.local as TRIGGER_DEV_SECRET_KEY first.');
  process.exitCode=1;
} else {
  configure({secretKey:key});
  try {
    const handle=await tasks.trigger<typeof alphaTraceHealthCheck>('alphatrace-health-check',{});
    console.log(JSON.stringify({task:'alphatrace-health-check',runId:handle.id}));
  } catch {
    console.error('The development task could not be triggered. Check the development key and running Trigger.dev worker.');
    process.exitCode=1;
  }
}
