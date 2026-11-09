-- Identify scheduled heartbeats separately from continuous local workers.
ALTER TABLE alphatrace.monitor_targets ADD COLUMN worker_mode text NOT NULL DEFAULT 'local'
 CHECK (worker_mode IN ('local','cloud'));
ALTER TABLE alphatrace.provider_targets ADD COLUMN worker_mode text NOT NULL DEFAULT 'local'
 CHECK (worker_mode IN ('local','cloud'));
