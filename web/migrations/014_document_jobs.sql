CREATE TABLE alphatrace.document_jobs (
 id uuid PRIMARY KEY DEFAULT gen_random_uuid(),owner_id uuid NOT NULL,command_id uuid NOT NULL,
 kind text NOT NULL CHECK(kind IN ('attach','check','tcs')),
 input jsonb NOT NULL,files jsonb NOT NULL,input_hash text NOT NULL,
 state text NOT NULL DEFAULT 'uploading' CHECK(state IN ('uploading','queued','running','succeeded','failed','expired')),
 result jsonb,message text,worker_run_id text,lease_until timestamptz,
 created_at timestamptz NOT NULL DEFAULT now(),expires_at timestamptz NOT NULL DEFAULT now()+interval '1 hour',
 cleanup_after timestamptz NOT NULL DEFAULT now()+interval '4 hours',cleaned_at timestamptz,finished_at timestamptz,
 UNIQUE(owner_id,command_id),CHECK(octet_length(input::text)<16384),CHECK(octet_length(files::text)<8192)
);
CREATE INDEX document_jobs_cleanup ON alphatrace.document_jobs(cleanup_after) WHERE cleaned_at IS NULL;
ALTER TABLE alphatrace.document_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE alphatrace.document_jobs FORCE ROW LEVEL SECURITY;
CREATE POLICY operator_only ON alphatrace.document_jobs FOR ALL TO alphatrace_app
 USING(EXISTS(SELECT 1 FROM alphatrace.research_operators WHERE user_id=NULLIF(current_setting('app.user_id',true),'')::uuid))
 WITH CHECK(EXISTS(SELECT 1 FROM alphatrace.research_operators WHERE user_id=NULLIF(current_setting('app.user_id',true),'')::uuid));
REVOKE ALL ON alphatrace.document_jobs FROM PUBLIC;
GRANT SELECT,INSERT ON alphatrace.document_jobs TO alphatrace_app;
GRANT UPDATE(state,result,message,worker_run_id,lease_until,cleaned_at,finished_at) ON alphatrace.document_jobs TO alphatrace_app;
