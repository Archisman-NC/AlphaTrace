import { defineConfig } from "@trigger.dev/sdk";
import { aptGet,syncEnvVars } from '@trigger.dev/build/extensions/core';
import { mkdir,copyFile } from 'node:fs/promises';
import { join } from 'node:path';
import { productionEnvironment } from './scripts/lib/trigger-environment';

export default defineConfig({
  project: "proj_azibgotclaclvmjiifqv",
  runtime: "node-22",
  logLevel: "log",
  // The max compute seconds a task is allowed to run. If the task run exceeds this duration, it will be stopped.
  // You can override this on an individual task.
  // See https://trigger.dev/docs/runs/max-duration
  maxDuration: 60,
  retries: {
    enabledInDev: false,
    default: {
      maxAttempts: 3,
      minTimeoutInMs: 1000,
      maxTimeoutInMs: 10000,
      factor: 2,
      randomize: true,
    },
  },
  dirs: ["./src/trigger"],
  legacyDevProcessCwdBehaviour:false,
  build:{extensions:[
    aptGet({packages:['python3','poppler-utils','ca-certificates']}),
    {name:'official-source-reader',async onBuildComplete(context,manifest){
      // Fail deployment before syncEnvVars can downgrade a configuration error to a warning.
      if(context.target==='deploy'&&manifest.environment==='prod')productionEnvironment(context.workingDir);
      const destination=join(manifest.outputPath,'research');
      await mkdir(destination,{recursive:true});
      for(const file of ['batch.py','sec_infy.py','tcs_workbook.py','sources.json'])await copyFile(join(context.workingDir,'../scripts/research',file),join(destination,file));
    }},
    syncEnvVars(({environment})=>environment==='prod'?productionEnvironment():[]),
  ]},
});
