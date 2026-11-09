-- Shared source monitoring. Only operators can schedule/claim/finish work.
CREATE TABLE alphatrace.monitor_targets (
 symbol text PRIMARY KEY CHECK(symbol IN ('TCS','INFY')),
 enabled boolean NOT NULL DEFAULT false,
 next_due_at timestamptz,
 worker_seen_at timestamptz,
 active_run_id uuid,
 lease_until timestamptz,
 attempt integer NOT NULL DEFAULT 0 CHECK(attempt BETWEEN 0 AND 2),
 CHECK ((active_run_id IS NULL) = (lease_until IS NULL))
);
CREATE TABLE alphatrace.monitor_runs (
 id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
 symbol text NOT NULL REFERENCES alphatrace.monitor_targets(symbol),
 started_at timestamptz NOT NULL DEFAULT now(), finished_at timestamptz,
 state text NOT NULL DEFAULT 'running' CHECK(state IN ('running','success','failed')),
 attempt integer NOT NULL CHECK(attempt BETWEEN 1 AND 2),
 message text NOT NULL DEFAULT 'Checking configured official documents.',
 failure_code text,
 draft_id uuid REFERENCES alphatrace.research_drafts(id),
 draft_created boolean NOT NULL DEFAULT false,
 CHECK ((state='running') = (finished_at IS NULL))
);
ALTER TABLE alphatrace.monitor_targets ADD CONSTRAINT monitor_active_run FOREIGN KEY(active_run_id) REFERENCES alphatrace.monitor_runs(id);
CREATE INDEX monitor_recent_runs ON alphatrace.monitor_runs(symbol,started_at DESC);
CREATE INDEX monitor_due_targets ON alphatrace.monitor_targets(next_due_at) WHERE enabled;

ALTER TABLE alphatrace.monitor_targets ENABLE ROW LEVEL SECURITY;
ALTER TABLE alphatrace.monitor_targets FORCE ROW LEVEL SECURITY;
ALTER TABLE alphatrace.monitor_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE alphatrace.monitor_runs FORCE ROW LEVEL SECURITY;
CREATE POLICY read_monitor_targets ON alphatrace.monitor_targets FOR SELECT TO alphatrace_app USING(NULLIF(current_setting('app.user_id',true),'') IS NOT NULL);
CREATE POLICY read_monitor_runs ON alphatrace.monitor_runs FOR SELECT TO alphatrace_app USING(NULLIF(current_setting('app.user_id',true),'') IS NOT NULL);
CREATE POLICY operator_monitor_targets ON alphatrace.monitor_targets FOR ALL TO alphatrace_app
 USING(EXISTS(SELECT 1 FROM alphatrace.research_operators WHERE user_id=NULLIF(current_setting('app.user_id',true),'')::uuid))
 WITH CHECK(EXISTS(SELECT 1 FROM alphatrace.research_operators WHERE user_id=NULLIF(current_setting('app.user_id',true),'')::uuid));
CREATE POLICY operator_monitor_runs ON alphatrace.monitor_runs FOR ALL TO alphatrace_app
 USING(EXISTS(SELECT 1 FROM alphatrace.research_operators WHERE user_id=NULLIF(current_setting('app.user_id',true),'')::uuid))
 WITH CHECK(EXISTS(SELECT 1 FROM alphatrace.research_operators WHERE user_id=NULLIF(current_setting('app.user_id',true),'')::uuid));
GRANT SELECT,INSERT,UPDATE ON alphatrace.monitor_targets,alphatrace.monitor_runs TO alphatrace_app;
REVOKE ALL ON alphatrace.monitor_targets,alphatrace.monitor_runs FROM PUBLIC;
INSERT INTO alphatrace.monitor_targets(symbol) VALUES('TCS'),('INFY');
