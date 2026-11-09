-- The migration connection is privileged; the application connection is not.
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'alphatrace_app') THEN
    CREATE ROLE alphatrace_app NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOBYPASSRLS;
  END IF;
END $$;
CREATE SCHEMA IF NOT EXISTS alphatrace;
REVOKE ALL ON SCHEMA alphatrace FROM PUBLIC;
GRANT USAGE ON SCHEMA alphatrace TO alphatrace_app;
CREATE TABLE alphatrace.workspaces (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_id uuid NOT NULL,
  name text NOT NULL CHECK (char_length(btrim(name)) BETWEEN 1 AND 80),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT one_workspace_per_owner UNIQUE (owner_id)
);
ALTER TABLE alphatrace.workspaces ENABLE ROW LEVEL SECURITY;
ALTER TABLE alphatrace.workspaces FORCE ROW LEVEL SECURITY;
CREATE POLICY own_workspace ON alphatrace.workspaces
  FOR ALL TO alphatrace_app
  USING (owner_id = NULLIF(current_setting('app.user_id', true), '')::uuid)
  WITH CHECK (owner_id = NULLIF(current_setting('app.user_id', true), '')::uuid);
GRANT SELECT, INSERT, UPDATE ON alphatrace.workspaces TO alphatrace_app;
REVOKE ALL ON alphatrace.workspaces FROM PUBLIC;
COMMENT ON TABLE alphatrace.workspaces IS 'Private workspaces; owner identity is verified server-side before a transaction-local RLS context is set.';
