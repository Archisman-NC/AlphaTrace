#!/usr/bin/env python3
"""Bounded, deterministic source checks. Emits drafts; never approves or publishes."""
import argparse
import copy
import datetime as dt
import hashlib
import html.parser
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(os.environ.get('ALPHATRACE_RESEARCH_ROOT', Path(__file__).resolve().parents[2]))
LIMIT = 20 * 1024 * 1024
HOSTS = {'TCS': 'www.tcs.com', 'INFY': 'www.infosys.com'}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('Redirect requires operator verification')


class TextOnly(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ('script', 'style') and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def clean(text):
    return re.sub(r'\s+', ' ', text).strip()


def source_url(url, symbol):
    u = urllib.parse.urlsplit(url)
    sec = symbol == 'INFY' and u.hostname == 'www.sec.gov' and re.fullmatch(r'/Archives/edgar/data/1067491/\d{18}/[a-zA-Z0-9_-]+\.html?', u.path)
    if u.scheme != 'https' or not (u.hostname == HOSTS[symbol] or sec) or u.username or u.password or u.port or u.query or u.fragment:
        raise ValueError('Source is outside the official allowlist')
    return url


def read_source(spec, symbol, snapshots):
    url = source_url(spec['url'], symbol)
    meta = {k: v for k, v in spec.items() if k not in ('assertions',)}
    if snapshots:
        file = snapshots / (spec['id'] + '.snapshot.json')
        if file.stat().st_size > LIMIT:
            raise ValueError('Snapshot exceeds size limit')
        snapshot = json.loads(file.read_text())
        if snapshot['url'] != url:
            raise ValueError('Snapshot URL does not match source')
        raw = snapshot['text'].encode('utf-8')
        retrieved = dt.datetime.fromisoformat(snapshot['retrievedAt'].replace('Z', '+00:00'))
        if retrieved.tzinfo is None or retrieved > dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=5):
            raise ValueError('Invalid snapshot retrieval time')
        # Replaying a snapshot must not pretend to be a fresh source check.
        if dt.datetime.now(dt.timezone.utc) - retrieved > dt.timedelta(days=7):
            raise ValueError('Snapshot is older than seven days; retrieve the source again')
        pages = {}
        for line in snapshot['text'].splitlines():
            m = re.match(r'L\d+@P(\d+)(?:-\d+)?:\s*(.*)', line)
            if m:
                pages.setdefault(int(m[1])+1, []).append(m[2])
        if spec['mediaType'] == 'application/pdf':
            if not pages:
                raise ValueError('PDF snapshot must retain zero-based web page markers')
            pages = {page: clean(' '.join(lines)) for page, lines in pages.items()}
        else:
            pages = {1: clean(re.sub(r'(?m)^L\d+:\s*', '', snapshot['text']))}
        meta.update(fetchedAt=snapshot['retrievedAt'], fingerprintKind='extracted_text', retrieval='operator_snapshot')
    else:
        opener = urllib.request.build_opener(NoRedirect)
        req = urllib.request.Request(url, headers={'User-Agent': 'AlphaTraceResearch/1.0 (bounded source check)'})
        with opener.open(req, timeout=30) as response:
            if response.headers.get_content_type() != spec['mediaType']:
                raise ValueError('Unexpected source content type')
            raw = response.read(LIMIT+1)
        if len(raw) > LIMIT:
            raise ValueError('Source exceeds size limit')
        if spec['mediaType'] == 'application/pdf':
            if not raw.startswith(b'%PDF-'):
                raise ValueError('Invalid PDF signature')
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'source.pdf'
                path.write_bytes(raw)
                info = subprocess.run(['pdfinfo', str(path)], check=True, capture_output=True, timeout=15).stdout.decode()
                count = re.search(r'^Pages:\s+(\d+)', info, re.M)
                if not count or not 1 <= int(count[1]) <= 100 or int(count[1]) != spec['pages']:
                    raise ValueError('PDF page count changed or exceeded the limit')
                output = subprocess.run(['pdftotext', '-layout', str(path), '-'], check=True, capture_output=True, timeout=30).stdout
                if len(output) > LIMIT:
                    raise ValueError('Extracted text exceeds size limit')
                pages = {i+1: clean(p) for i, p in enumerate(output.decode().split('\f'))}
        else:
            parser = TextOnly()
            parser.feed(raw.decode('utf-8'))
            pages = {1: clean(' '.join(parser.parts))}
        meta.update(fetchedAt=now(), fingerprintKind='document_bytes', retrieval='direct')
    meta.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    joined = ' '.join(pages.values())
    for assertion in spec['assertions']:
        if not re.search(assertion, joined):
            raise ValueError(f"{spec['id']}: period, basis or source structure changed")
    cache = ROOT / '.local/research/cache'
    cache.mkdir(parents=True, exist_ok=True)
    (cache / (meta['sha256'] + '.source')).write_bytes(raw)
    return meta, pages


def extract_point(point, documents):
    result = copy.deepcopy(point)
    rule = result.pop('extract')
    page = documents[result['sourceId']].get(result['page'] or 1, '')
    matches = re.findall(rule['pattern'], page)
    if len(matches) != 1:
        raise ValueError(f"{result['sourceId']}: expected one unambiguous metric row, found {len(matches)}")
    match = matches[0]
    value = match[rule.get('group', 1)-1] if isinstance(match, tuple) else match
    result['value'] = value.replace(',', '')
    if not re.fullmatch(r'-?\d{1,15}(?:\.\d{1,6})?', result['value']):
        raise ValueError('Metric extraction did not produce an exact decimal')
    return result


def build_draft(spec, snapshots=None):
    draft = copy.deepcopy(spec)
    documents, sources = {}, []
    for source in spec['sources']:
        meta, pages = read_source(source, spec['symbol'], snapshots)
        sources.append(meta)
        documents[source['id']] = pages
    draft['sources'] = sources
    for comparison in draft['comparisons']:
        for side in ('previous', 'current'):
            comparison[side] = extract_point(comparison[side], documents)
    # Cross-check selected current values in a second official document.
    for assertion in draft.pop('crossChecks', []):
        value = extract_point(assertion['point'], documents)['value']
        metric = next(c for c in draft['comparisons'] if c['metric'] == assertion['metric'])
        if value != metric[assertion['side']]['value']:
            raise ValueError('Official documents disagree; operator reconciliation required')
    checked = min(s['fetchedAt'] for s in sources)
    message = ('Operator-supplied source snapshots validated; automatic retrieval is not verified.' if snapshots else 'Configured official documents retrieved and extraction checks passed. This does not discover new releases.')
    return {'draft': draft, 'check': {'symbol': draft['symbol'], 'checkedAt': checked, 'status': 'success', 'message': message, 'fingerprints': [s['sha256'] for s in sources]}}


def submit(bundle, api, token):
    u = urllib.parse.urlsplit(api)
    if u.username or u.password or u.query or u.fragment or (u.scheme != 'https' and not (u.scheme == 'http' and u.hostname in ('localhost', '127.0.0.1'))):
        raise ValueError('Use HTTPS or a loopback API origin')
    if len(token) < 32:
        raise ValueError('Set ALPHATRACE_INGEST_TOKEN in the environment')
    request = urllib.request.Request(api.rstrip('/')+'/api/research/ingest', data=json.dumps(bundle).encode(), headers={'Content-Type': 'application/json', 'Authorization': 'Bearer '+token}, method='POST')
    with urllib.request.build_opener(NoRedirect).open(request, timeout=30) as response:
        return json.loads(response.read(16384))


def failure_bundle(symbol, exc):
    """Stable failure reasons, without raw response bodies, URLs or credentials."""
    if isinstance(exc, urllib.error.HTTPError):
        code = 'http_' + str(exc.code)
        message = ('The official source denied automated access (HTTP %s). Manual source review is required.' % exc.code
                   if exc.code in (401, 403) else 'The official source returned HTTP %s. No fresh evidence was confirmed.' % exc.code)
    elif isinstance(exc, (TimeoutError, socket.timeout, subprocess.TimeoutExpired)):
        code, message = 'timeout', 'The source check timed out. No fresh evidence was confirmed.'
    elif isinstance(exc, urllib.error.URLError):
        code, message = 'network', 'The official source could not be reached. No fresh evidence was confirmed.'
    elif isinstance(exc, FileNotFoundError):
        code, message = 'dependency', 'A required source-processing tool or input is unavailable. Operator attention is required.'
    else:
        code, message = 'validation', 'Source dates, structure or extracted values could not be validated. Manual source review is required.'
    return {'check': {'symbol': symbol, 'checkedAt': now(), 'status': 'failed', 'message': message, 'fingerprints': []}, 'failureCode': code}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbol', choices=sorted(HOSTS))
    parser.add_argument('--snapshots', type=Path, help='Operator-supplied source snapshots; preserves original retrieval times')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--api', default='http://127.0.0.1:3000')
    parser.add_argument('--output-dir', type=Path, help='Private per-run output directory for the monitoring worker')
    args = parser.parse_args()
    specs = json.loads((Path(__file__).parent/'sources.json').read_text())
    output = args.output_dir or ROOT/'.local/research/output'
    output.mkdir(parents=True, exist_ok=True)
    failed = False
    for spec in specs:
        if args.symbol and spec['symbol'] != args.symbol:
            continue
        try:
            if spec['symbol'] == 'INFY' and not args.snapshots:
                from sec_infy import build
                bundle = build()
            else:
                bundle = build_draft(spec, args.snapshots)
        except (ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
            failed = True
            bundle = failure_bundle(spec['symbol'], exc.__cause__ or exc)
            if getattr(exc, 'discovery', None):
                bundle['discovery'] = exc.discovery
                bundle['check']['message'] = str(exc)
        temporary = output/(spec['symbol']+'.json.tmp')
        temporary.write_text(json.dumps(bundle, indent=2)+'\n')
        temporary.replace(output/(spec['symbol']+'.json'))
        print(spec['symbol']+': '+bundle['check']['status']+'; '+('unpublished draft prepared' if 'draft' in bundle else 'failure recorded'))
        if args.submit:
            result = submit(bundle, args.api, os.environ.get('ALPHATRACE_INGEST_TOKEN', ''))
            print(spec['symbol']+': '+('new draft' if result.get('created') else 'existing draft / check only'))
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
