CREATE TABLE alphatrace.email_preferences (
 owner_id uuid PRIMARY KEY REFERENCES alphatrace.workspaces(owner_id) ON DELETE CASCADE,
 version integer NOT NULL DEFAULT 0 CHECK(version>=0), enabled boolean NOT NULL DEFAULT false,
 reminders boolean NOT NULL DEFAULT true, enabled_since timestamptz,
 suppressed boolean NOT NULL DEFAULT false, updated_at timestamptz NOT NULL DEFAULT now(),
 CHECK(NOT enabled OR (enabled_since IS NOT NULL AND NOT suppressed))
);
CREATE TABLE alphatrace.email_deliveries (
 id uuid PRIMARY KEY, owner_id uuid NOT NULL REFERENCES alphatrace.workspaces(owner_id) ON DELETE CASCADE,
 digest_day date NOT NULL, preference_version integer NOT NULL,
 recipient text NOT NULL, payload jsonb NOT NULL, items jsonb NOT NULL,
 unsubscribe_hash text NOT NULL UNIQUE CHECK(unsubscribe_hash ~ '^[a-f0-9]{64}$'),
 state text NOT NULL DEFAULT 'pending' CHECK(state IN ('pending','sending','retry','accepted','failed','uncertain','cancelled')),
 attempts integer NOT NULL DEFAULT 0, first_attempt_at timestamptz, next_attempt_at timestamptz NOT NULL DEFAULT now(),
 lease_id uuid, lease_until timestamptz, provider_id text UNIQUE,
 message text NOT NULL DEFAULT 'Waiting for delivery.', created_at timestamptz NOT NULL DEFAULT now(), accepted_at timestamptz,
 delivered_at timestamptz, bounced_at timestamptz, complained_at timestamptz,
 UNIQUE(owner_id,digest_day), UNIQUE(owner_id,id)
);
CREATE TABLE alphatrace.email_delivery_items (
 owner_id uuid NOT NULL, delivery_id uuid NOT NULL, event_key text NOT NULL,
 PRIMARY KEY(owner_id,event_key),
 FOREIGN KEY(owner_id,delivery_id) REFERENCES alphatrace.email_deliveries(owner_id,id) ON DELETE CASCADE
);
CREATE TABLE alphatrace.email_webhook_events (
 event_id text PRIMARY KEY, provider_id text NOT NULL, event_type text NOT NULL, received_at timestamptz NOT NULL DEFAULT now()
);
DO $$ DECLARE t text; BEGIN
 FOREACH t IN ARRAY ARRAY['email_preferences','email_deliveries','email_delivery_items'] LOOP
  EXECUTE format('ALTER TABLE alphatrace.%I ENABLE ROW LEVEL SECURITY',t);
  EXECUTE format('ALTER TABLE alphatrace.%I FORCE ROW LEVEL SECURITY',t);
  EXECUTE format('CREATE POLICY own_email_data ON alphatrace.%I FOR ALL TO alphatrace_app USING(owner_id=NULLIF(current_setting(''app.user_id'',true),'''')::uuid) WITH CHECK(owner_id=NULLIF(current_setting(''app.user_id'',true),'''')::uuid)',t);
  EXECUTE format('REVOKE ALL ON alphatrace.%I FROM PUBLIC',t);
 END LOOP;
END $$;
GRANT SELECT,INSERT,UPDATE ON alphatrace.email_preferences,alphatrace.email_deliveries TO alphatrace_app;
GRANT SELECT,INSERT ON alphatrace.email_delivery_items TO alphatrace_app;
ALTER TABLE alphatrace.email_webhook_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE alphatrace.email_webhook_events FORCE ROW LEVEL SECURITY;
REVOKE ALL ON alphatrace.email_webhook_events FROM PUBLIC,alphatrace_app;

-- Unauthenticated access can only disable emails for a possession token.
-- GET requests never call this function; tokens are generated from 32 random bytes.
CREATE FUNCTION alphatrace.unsubscribe_email(token_hash text) RETURNS void
 LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
 DECLARE target uuid;
 BEGIN
  SELECT owner_id INTO target FROM alphatrace.email_deliveries WHERE unsubscribe_hash=token_hash;
  IF target IS NULL THEN RETURN; END IF;
  UPDATE alphatrace.email_preferences SET enabled=false,version=version+1,updated_at=now() WHERE owner_id=target;
  UPDATE alphatrace.email_deliveries SET state='cancelled',message='Email preferences were disabled.'
   WHERE owner_id=target AND state IN ('pending','retry','sending');
 END $$;
REVOKE ALL ON FUNCTION alphatrace.unsubscribe_email(text) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION alphatrace.unsubscribe_email(text) TO alphatrace_app;

-- The HTTP handler verifies the raw Resend/Svix signature before invoking this
-- narrow function. It cannot read addresses, content, or arbitrary user data.
CREATE FUNCTION alphatrace.record_email_event(event_key text,email_id text,kind text,event_time timestamptz) RETURNS boolean
 LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
 DECLARE target uuid; delivery uuid;
 BEGIN
  IF kind NOT IN ('email.delivered','email.bounced','email.complained','email.failed','email.suppressed') THEN RETURN true; END IF;
  SELECT owner_id,id INTO target,delivery FROM alphatrace.email_deliveries WHERE provider_id=email_id;
  IF target IS NULL THEN RETURN false; END IF;
  -- Same preference-first lock order as dispatch and unsubscribe.
  PERFORM owner_id FROM alphatrace.email_preferences WHERE owner_id=target FOR UPDATE;
  INSERT INTO alphatrace.email_webhook_events(event_id,provider_id,event_type) VALUES(event_key,email_id,kind) ON CONFLICT DO NOTHING;
  IF NOT FOUND THEN RETURN true; END IF;
  UPDATE alphatrace.email_deliveries SET
   delivered_at=CASE WHEN kind='email.delivered' THEN COALESCE(delivered_at,event_time) ELSE delivered_at END,
   bounced_at=CASE WHEN kind IN ('email.bounced','email.suppressed') THEN COALESCE(bounced_at,event_time) ELSE bounced_at END,
   complained_at=CASE WHEN kind='email.complained' THEN COALESCE(complained_at,event_time) ELSE complained_at END,
   state=CASE WHEN kind='email.failed' AND delivered_at IS NULL THEN 'failed' ELSE state END,
   message=CASE WHEN kind='email.failed' AND delivered_at IS NULL THEN 'Resend reported a delivery failure.' ELSE message END
   WHERE id=delivery;
  IF kind IN ('email.bounced','email.complained','email.suppressed') THEN
   UPDATE alphatrace.email_preferences SET enabled=false,suppressed=true,version=version+1,updated_at=now() WHERE owner_id=target;
   UPDATE alphatrace.email_deliveries SET state='cancelled',message='Delivery stopped after a bounce or complaint.'
    WHERE owner_id=target AND state IN ('pending','retry','sending');
  END IF;
  RETURN true;
 END $$;
REVOKE ALL ON FUNCTION alphatrace.record_email_event(text,text,text,timestamptz) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION alphatrace.record_email_event(text,text,text,timestamptz) TO alphatrace_app;
