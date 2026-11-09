-- Database-owned five-seat beta, with capability links stored only as hashes.
CREATE TABLE alphatrace.beta_configuration (singleton boolean PRIMARY KEY DEFAULT true CHECK(singleton), initialized boolean NOT NULL DEFAULT false);
INSERT INTO alphatrace.beta_configuration VALUES(true,false);
CREATE TABLE alphatrace.beta_members (
 id uuid PRIMARY KEY, user_id uuid NOT NULL UNIQUE, email text UNIQUE,
 state text NOT NULL CHECK(state IN ('invited','active','revoked','closed')),
 token_hash text UNIQUE CHECK(token_hash IS NULL OR token_hash ~ '^[0-9a-f]{64}$'),
 expires_at timestamptz, version integer NOT NULL DEFAULT 1,
 created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(), activated_at timestamptz,
 CHECK(email IS NULL OR (email=lower(btrim(email)) AND length(email)<=254)),
 CHECK(state<>'invited' OR (token_hash IS NOT NULL AND expires_at IS NOT NULL))
);
CREATE TABLE alphatrace.beta_events (
 id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, member_id uuid NOT NULL REFERENCES alphatrace.beta_members(id),
 actor_id uuid, action text NOT NULL, created_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE alphatrace.beta_configuration ENABLE ROW LEVEL SECURITY;
ALTER TABLE alphatrace.beta_configuration FORCE ROW LEVEL SECURITY;
ALTER TABLE alphatrace.beta_members ENABLE ROW LEVEL SECURITY;
ALTER TABLE alphatrace.beta_members FORCE ROW LEVEL SECURITY;
ALTER TABLE alphatrace.beta_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE alphatrace.beta_events FORCE ROW LEVEL SECURITY;
REVOKE ALL ON alphatrace.beta_configuration,alphatrace.beta_members,alphatrace.beta_events FROM PUBLIC,alphatrace_app;

CREATE FUNCTION alphatrace.beta_access_status() RETURNS text LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
 SELECT CASE WHEN NOT (SELECT initialized FROM alphatrace.beta_configuration) THEN 'legacy'
 WHEN EXISTS(SELECT 1 FROM alphatrace.account_closures WHERE owner_id=NULLIF(current_setting('app.user_id',true),'')::uuid) THEN 'closed'
 ELSE coalesce((SELECT state FROM alphatrace.beta_members WHERE user_id=NULLIF(current_setting('app.user_id',true),'')::uuid),'missing') END;
$$;
CREATE FUNCTION alphatrace.beta_require_operator() RETURNS void LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
 BEGIN
 IF NOT EXISTS(SELECT 1 FROM alphatrace.research_operators WHERE user_id=NULLIF(current_setting('app.user_id',true),'')::uuid)
 OR alphatrace.beta_access_status() NOT IN ('legacy','active') THEN RAISE EXCEPTION 'Operator access required' USING ERRCODE='42501'; END IF;
 END $$;
-- Only an existing operator may perform the one-time migration of the old allowlist.
CREATE FUNCTION alphatrace.beta_initialize(profiles jsonb) RETURNS void LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
 DECLARE p jsonb; target uuid; actor uuid:=NULLIF(current_setting('app.user_id',true),'')::uuid;
 BEGIN
 PERFORM alphatrace.beta_require_operator();PERFORM pg_advisory_xact_lock(67241124);
 IF (SELECT initialized FROM alphatrace.beta_configuration) THEN RAISE EXCEPTION 'Already initialized' USING ERRCODE='23505'; END IF;
 IF jsonb_typeof(profiles)<>'array' OR jsonb_array_length(profiles) NOT BETWEEN 1 AND 5 THEN RAISE EXCEPTION 'Choose one to five existing users'; END IF;
 FOR p IN SELECT * FROM jsonb_array_elements(profiles) LOOP
  target:=(p->>'id')::uuid;
  IF length(p->>'email') NOT BETWEEN 3 AND 254 OR p->>'email' NOT LIKE '%@%' OR EXISTS(SELECT 1 FROM alphatrace.account_closures WHERE owner_id=target) THEN RAISE EXCEPTION 'Invalid existing account'; END IF;
  INSERT INTO alphatrace.beta_members(id,user_id,email,state,activated_at) VALUES(gen_random_uuid(),target,lower(btrim(p->>'email')),'active',now());
 END LOOP;
 IF NOT EXISTS(SELECT 1 FROM alphatrace.beta_members WHERE user_id=actor) THEN RAISE EXCEPTION 'Include the enrolled operator'; END IF;
 UPDATE alphatrace.beta_configuration SET initialized=true;
 END $$;
CREATE FUNCTION alphatrace.beta_people() RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
 DECLARE result jsonb;
 BEGIN
 PERFORM alphatrace.beta_require_operator();
 SELECT jsonb_build_object('initialized',(SELECT initialized FROM alphatrace.beta_configuration),'limit',5,'checkedAt',now(),
 'used',(SELECT count(*) FROM alphatrace.beta_members WHERE state='active' OR (state='invited' AND expires_at>now())),
 'people',coalesce((SELECT jsonb_agg(row_to_json(t) ORDER BY t."createdAt" DESC) FROM (
 SELECT m.id,m.email,m.version,CASE WHEN m.state='invited' AND m.expires_at<=now() THEN 'expired' ELSE m.state END AS state,
 m.created_at AS "createdAt",m.expires_at AS "expiresAt",m.activated_at AS "activatedAt",
 EXISTS(SELECT 1 FROM alphatrace.research_operators WHERE user_id=m.user_id) AS "isOperator",
 EXISTS(SELECT 1 FROM alphatrace.user_onboarding WHERE owner_id=m.user_id AND status='completed') AS "setupCompleted",
 EXISTS(SELECT 1 FROM alphatrace.review_events WHERE owner_id=m.user_id) AS "firstResponseSaved"
 FROM alphatrace.beta_members m ORDER BY m.created_at DESC LIMIT 100) t),'[]'::jsonb)) INTO result;
 RETURN result;
 END $$;
CREATE FUNCTION alphatrace.beta_invite(member uuid,address text,hash text) RETURNS void LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
 DECLARE existing alphatrace.beta_members; actor uuid:=NULLIF(current_setting('app.user_id',true),'')::uuid;
 BEGIN
 PERFORM alphatrace.beta_require_operator();PERFORM pg_advisory_xact_lock(67241124);
 IF NOT (SELECT initialized FROM alphatrace.beta_configuration) THEN RAISE EXCEPTION 'Initialize beta access first'; END IF;
 IF hash !~ '^[0-9a-f]{64}$' OR address IS NULL OR length(address) NOT BETWEEN 3 AND 254 OR address NOT LIKE '%@%' THEN RAISE EXCEPTION 'Invalid invitation'; END IF;
 SELECT * INTO existing FROM alphatrace.beta_members WHERE id=member;
 IF FOUND THEN
  IF existing.email=address AND existing.token_hash=hash AND existing.state='invited' AND existing.expires_at>now() THEN RETURN; END IF;
  RAISE EXCEPTION 'Invitation changed. Refresh People.' USING ERRCODE='23505';
 END IF;
 IF EXISTS(SELECT 1 FROM alphatrace.beta_members WHERE email=address) THEN RAISE EXCEPTION 'This email already has a record. Use its existing row.' USING ERRCODE='23505'; END IF;
 IF (SELECT count(*) FROM alphatrace.beta_members WHERE state='active' OR (state='invited' AND expires_at>now()))>=5 THEN RAISE EXCEPTION 'All five beta seats are reserved.' USING ERRCODE='23505'; END IF;
 INSERT INTO alphatrace.beta_members(id,user_id,email,state,token_hash,expires_at) VALUES(member,gen_random_uuid(),address,'invited',hash,now()+interval '3 days');
 INSERT INTO alphatrace.beta_events(member_id,actor_id,action) VALUES(member,actor,'invited');
 END $$;
CREATE FUNCTION alphatrace.beta_manage(member uuid,expected integer,action text,hash text DEFAULT NULL) RETURNS void LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
 DECLARE m alphatrace.beta_members; actor uuid:=NULLIF(current_setting('app.user_id',true),'')::uuid;
 BEGIN
 PERFORM alphatrace.beta_require_operator();PERFORM pg_advisory_xact_lock(67241124);
 SELECT * INTO m FROM alphatrace.beta_members WHERE id=member;
 IF FOUND THEN
  PERFORM pg_advisory_xact_lock(hashtext('account:'||m.user_id::text));
  SELECT * INTO m FROM alphatrace.beta_members WHERE id=member FOR UPDATE;
 END IF;
 IF NOT FOUND THEN RAISE EXCEPTION 'Person not found' USING ERRCODE='23505'; END IF;
 IF m.user_id=actor OR EXISTS(SELECT 1 FROM alphatrace.research_operators WHERE user_id=m.user_id) THEN RAISE EXCEPTION 'Operator access cannot be changed here' USING ERRCODE='23505'; END IF;
 IF m.state='closed' THEN RAISE EXCEPTION 'This account is closed' USING ERRCODE='23505'; END IF;
 IF action='revoke' AND m.state='revoked' AND m.version=expected+1 THEN RETURN; END IF;
 IF action='reissue' AND m.state='invited' AND m.token_hash=hash AND m.expires_at>now() AND m.version=expected+1 THEN RETURN; END IF;
 IF m.version<>expected THEN RAISE EXCEPTION 'Access changed. Refresh People.' USING ERRCODE='23505'; END IF;
 IF action='revoke' THEN
  PERFORM pg_advisory_xact_lock(hashtext('account:'||m.user_id::text));
  UPDATE alphatrace.beta_members SET state='revoked',token_hash=NULL,version=version+1,updated_at=now() WHERE id=member;
 ELSIF action='reissue' THEN
  IF hash IS NULL OR hash !~ '^[0-9a-f]{64}$' OR m.state='active' THEN RAISE EXCEPTION 'Only pending or revoked invitations can be issued again' USING ERRCODE='23505'; END IF;
  IF (SELECT count(*) FROM alphatrace.beta_members WHERE id<>member AND (state='active' OR (state='invited' AND expires_at>now())))>=5 THEN RAISE EXCEPTION 'All five beta seats are reserved.' USING ERRCODE='23505'; END IF;
  UPDATE alphatrace.beta_members SET state='invited',token_hash=hash,expires_at=now()+interval '3 days',version=version+1,updated_at=now() WHERE id=member;
 ELSE RAISE EXCEPTION 'Unknown access action'; END IF;
 INSERT INTO alphatrace.beta_events(member_id,actor_id,action) VALUES(member,actor,action);
 END $$;
-- Anonymous callers can inspect only a high-entropy invitation they possess.
CREATE FUNCTION alphatrace.beta_invitation(hash text,lock_row boolean DEFAULT false) RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
 DECLARE m alphatrace.beta_members;
 BEGIN
 IF hash IS NULL OR hash !~ '^[0-9a-f]{64}$' THEN RETURN NULL; END IF;
 IF lock_row THEN
  PERFORM pg_advisory_xact_lock(67241124);
  SELECT * INTO m FROM alphatrace.beta_members WHERE token_hash=hash FOR UPDATE;
 ELSE SELECT * INTO m FROM alphatrace.beta_members WHERE token_hash=hash; END IF;
 IF m.id IS NULL OR m.state<>'invited' OR m.expires_at<=now() OR NOT (SELECT initialized FROM alphatrace.beta_configuration)
 OR EXISTS(SELECT 1 FROM alphatrace.account_closures WHERE owner_id=m.user_id) THEN RETURN NULL; END IF;
 RETURN jsonb_build_object('id',m.id,'userId',m.user_id,'email',m.email,'expiresAt',m.expires_at,'existing',m.activated_at IS NOT NULL);
 END $$;
CREATE FUNCTION alphatrace.beta_activate(hash text,verified_user uuid) RETURNS void LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
 DECLARE m jsonb; member uuid;
 BEGIN
 m:=alphatrace.beta_invitation(hash,true);IF m IS NULL THEN RAISE EXCEPTION 'Invitation expired, used or revoked' USING ERRCODE='23505'; END IF;
 member:=(m->>'id')::uuid;
 IF EXISTS(SELECT 1 FROM alphatrace.account_closures WHERE owner_id=verified_user)
 OR EXISTS(SELECT 1 FROM alphatrace.research_operators WHERE user_id=verified_user)
 OR EXISTS(SELECT 1 FROM alphatrace.beta_members WHERE user_id=verified_user AND id<>member) THEN RAISE EXCEPTION 'This account cannot accept this invitation' USING ERRCODE='23505'; END IF;
 IF (m->>'existing')::boolean AND (m->>'userId')::uuid<>verified_user THEN RAISE EXCEPTION 'Use the invited account' USING ERRCODE='23505'; END IF;
 UPDATE alphatrace.beta_members SET user_id=verified_user,state='active',token_hash=NULL,activated_at=coalesce(activated_at,now()),updated_at=now(),version=version+1 WHERE id=member;
 INSERT INTO alphatrace.beta_events(member_id,actor_id,action) VALUES(member,verified_user,'activated');
 END $$;
CREATE FUNCTION alphatrace.beta_active_users() RETURNS SETOF uuid LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
 BEGIN PERFORM alphatrace.beta_require_operator();RETURN QUERY SELECT user_id FROM alphatrace.beta_members WHERE state='active';END $$;
CREATE FUNCTION alphatrace.beta_close_member() RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
 BEGIN
 UPDATE alphatrace.beta_members SET state='closed',email=NULL,token_hash=NULL,updated_at=now(),version=version+1 WHERE user_id=NEW.owner_id;
 RETURN NEW;
 END $$;
CREATE TRIGGER beta_close_member AFTER INSERT ON alphatrace.account_closures FOR EACH ROW EXECUTE FUNCTION alphatrace.beta_close_member();
REVOKE ALL ON FUNCTION alphatrace.beta_access_status(),alphatrace.beta_require_operator(),alphatrace.beta_initialize(jsonb),alphatrace.beta_people(),alphatrace.beta_invite(uuid,text,text),alphatrace.beta_manage(uuid,integer,text,text),alphatrace.beta_invitation(text,boolean),alphatrace.beta_activate(text,uuid),alphatrace.beta_active_users(),alphatrace.beta_close_member() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION alphatrace.beta_access_status(),alphatrace.beta_initialize(jsonb),alphatrace.beta_people(),alphatrace.beta_invite(uuid,text,text),alphatrace.beta_manage(uuid,integer,text,text),alphatrace.beta_invitation(text,boolean),alphatrace.beta_activate(text,uuid),alphatrace.beta_active_users() TO alphatrace_app;
