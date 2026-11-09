-- Discovery is separate from extraction success. Failed extraction can still
-- preserve the latest observed filing; a later feed outage cannot erase it.
ALTER TABLE alphatrace.monitor_runs ADD COLUMN discovery jsonb
 CHECK(discovery IS NULL OR (jsonb_typeof(discovery)='object' AND octet_length(discovery::text)<=4096));
