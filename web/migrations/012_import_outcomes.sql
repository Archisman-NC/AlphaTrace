-- Per-attempt lineage is recorded atomically with its saved response and comparison.
ALTER TABLE alphatrace.provider_runs
 ADD COLUMN detection_result text NOT NULL DEFAULT 'not_recorded'
  CHECK(detection_result IN ('not_recorded','baseline','changed','unchanged','stale')),
 ADD COLUMN review_id uuid REFERENCES alphatrace.provider_reviews(id),
 ADD COLUMN worker_run_id text CHECK(worker_run_id ~ '^run_[a-zA-Z0-9]{10,80}$'),
 ADD CONSTRAINT provider_detection_review CHECK((detection_result IN ('baseline','changed'))=(review_id IS NOT NULL));
CREATE INDEX provider_runs_cloud_batch ON alphatrace.provider_runs(worker_run_id,started_at DESC) WHERE worker_run_id IS NOT NULL;
