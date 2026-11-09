"""Infosys SEC discovery and table extraction. No search service or model dependency."""
import datetime as dt
import hashlib
from html.parser import HTMLParser
import json
import re
import time
import urllib.parse
import urllib.request

from batch import LIMIT, NoRedirect, ROOT, clean, now, source_url

FEED = 'https://data.sec.gov/submissions/CIK0001067491.json'
ARCHIVE = 'https://www.sec.gov/Archives/edgar/data/1067491/'


class DiscoveryError(ValueError):
    def __init__(self, message, discovery=None):
        super().__init__(message)
        self.discovery = discovery


class Tables(HTMLParser):
    """Retain table boundaries/cells, headings and links; ignore executable content."""
    def __init__(self):
        super().__init__()
        self.tables, self.stack, self.parts, self.links = [], [], [], []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.skip += 1
        if self.skip:
            return
        if tag == 'a' and dict(attrs).get('href'):
            self.links.append(dict(attrs)['href'])
        if tag == 'table':
            self.stack.append({'before': clean(' '.join(self.parts))[-1000:], 'rows': [], 'row': None, 'cell': None})
        if self.stack:
            if tag == 'tr':
                self.stack[-1]['row'] = []
            if tag in ('td', 'th'):
                self.stack[-1]['cell'] = []

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)
            if self.stack and self.stack[-1]['cell'] is not None:
                self.stack[-1]['cell'].append(data)

    def handle_endtag(self, tag):
        if tag in ('script', 'style') and self.skip:
            self.skip -= 1
            return
        if self.skip or not self.stack:
            return
        table = self.stack[-1]
        if tag in ('td', 'th') and table['cell'] is not None and table['row'] is not None:
            table['row'].append(clean(' '.join(table['cell'])))
            table['cell'] = None
        if tag == 'tr' and table['row'] is not None:
            table['rows'].append(table['row'])
            table['row'] = None
        if tag == 'table':
            self.tables.append(self.stack.pop())

    @property
    def text(self):
        return clean(' '.join(self.parts))


def fetch(url, media):
    if url != FEED:
        source_url(url, 'INFY')
    # At most one request per second in this sequential, four-document adapter.
    time.sleep(1)
    request = urllib.request.Request(url, headers={'User-Agent': 'AlphaTraceResearch/1.0 (bounded Infosys filing monitor)'})
    with urllib.request.build_opener(NoRedirect).open(request, timeout=25) as response:
        if response.headers.get_content_type() != media:
            raise ValueError('Unexpected SEC response type')
        raw = response.read(LIMIT + 1)
    if len(raw) > LIMIT:
        raise ValueError('SEC response exceeded size limit')
    digest = hashlib.sha256(raw).hexdigest()
    cache = ROOT / '.local/research/cache'
    cache.mkdir(parents=True, exist_ok=True)
    (cache / (digest + '.source')).write_bytes(raw)
    return raw, now(), digest


def calendar_date(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        raise ValueError('Invalid filing date')
    date = dt.date.fromisoformat(value)
    if date > dt.datetime.now(dt.timezone.utc).date():
        raise ValueError('Future filing date')
    return date


def discover(data, stamp, fingerprint):
    if str(data.get('cik', '')).lstrip('0') != '1067491' or 'INFY' not in data.get('tickers', []) or 'infosys' not in data.get('name', '').lower():
        raise ValueError('SEC issuer identity mismatch')
    recent = data['filings']['recent']
    keys = ['accessionNumber', 'form', 'filingDate', 'reportDate', 'primaryDocument', 'primaryDocDescription']
    size = len(recent['accessionNumber'])
    if not size or size > 10000 or any(len(recent[k]) != size for k in keys):
        raise ValueError('SEC feed structure changed')
    # Recent results only. This is discovery, not a historical backfill/crawl.
    candidates = []
    for i in range(min(size, 300)):
        if recent['form'][i] not in ('6-K', '6-K/A') or 'RESULTS OF OPERATIONS' not in recent['primaryDocDescription'][i].upper():
            continue
        filed = calendar_date(recent['filingDate'][i])
        if (dt.datetime.now(dt.timezone.utc).date() - filed).days > 180:
            continue
        accession, document = recent['accessionNumber'][i], recent['primaryDocument'][i]
        if not re.fullmatch(r'\d{10}-\d{2}-\d{6}', accession) or not re.fullmatch(r'[a-zA-Z0-9_-]+\.htm[l]?', document):
            raise ValueError('Invalid SEC accession or document path')
        candidates.append({'accession': accession, 'filedOn': filed.isoformat(), 'periodEnd': recent['reportDate'][i],
                           'url': ARCHIVE + accession.replace('-', '') + '/' + document})
    if not candidates:
        raise ValueError('No recent results filing found; cannot assert current coverage')
    selected = max(candidates, key=lambda c: (c['filedOn'], c['accession']))
    # Keep the newest filing even if its extraction is unsupported. Never silently use an older result.
    return {**selected, 'checkedAt': stamp, 'feedUrl': FEED, 'feedSha256': fingerprint, 'candidateCount': len(candidates)}


def one(items, label):
    if len(items) != 1:
        raise ValueError('Expected one unambiguous ' + label)
    return items[0]


def exhibit_links(cover, filing):
    parser = Tables()
    parser.feed(cover)
    period = calendar_date(filing['periodEnd'])
    if (period.month, period.day) not in ((3, 31), (6, 30), (9, 30), (12, 31)) or period.isoformat() > filing['filedOn']:
        raise ValueError('Unsupported or inconsistent result period')
    full_date = period.strftime('%B') + f' {period.day}, {period.year}'
    if 'Infosys Limited' not in parser.text or full_date not in parser.text:
        raise ValueError('Filing issuer/period assertion failed')
    rows = [r for table in parser.tables for r in table['rows'] if len(r) == 2 and re.fullmatch(r'99\.\d+', r[0])]
    facts = one([r[0] for r in rows if r[1].lower() == 'fact sheet'], 'fact-sheet exhibit')
    results = one([r[0] for r in rows if 'consolidated' in r[1].lower() and 'financial statements' in r[1].lower()
                   and 'US Dollars' in r[1] and ('IFRS' in r[1] or 'International Financial Reporting Standards' in r[1])], 'USD IFRS exhibit')
    # Map the numbered exhibit only to an actual link present in this accession.
    urls = {}
    for key, number in [('facts', facts), ('results', results)]:
        filename = 'exv99w' + number.split('.')[1].zfill(2) + '.htm'
        link = one(list(set(h for h in parser.links if h == filename)), 'exhibit link')
        urls[key] = urllib.parse.urljoin(filing['url'], link)
    return urls


def decimal(value):
    if not re.fullmatch(r'-?(?:\d{1,3}(?:,\d{3})+|\d{1,15})(?:\.\d{1,6})?', value):
        raise ValueError('Missing or unsupported exact financial value')
    return value.replace(',', '')


def extract(facts_html, results_html, period_end):
    facts, results = Tables(), Tables()
    facts.feed(facts_html)
    results.feed(results_html)
    end = calendar_date(period_end)
    previous = (end.replace(day=1) - dt.timedelta(days=1)).replace(day=1) - dt.timedelta(days=1)
    previous = previous.replace(day=1) - dt.timedelta(days=1)
    # Dates come from table headers, never a guessed column offset across arbitrary tables.
    short = lambda d: d.strftime('%b') + f' {d.day}, {d.year}'
    prior_year = end.replace(year=end.year - 1)
    header = ['Particulars', short(end), short(prior_year), 'Growth % YoY', short(previous), 'Growth % QoQ']
    table = one([t for t in facts.tables if t['rows'] and t['rows'][0] == header
                 and re.search(r'Consolidated Statement of Comprehensive Income for three months ended', t['before'])
                 and re.search(r'IFRS Financial Statement.*In US \$ million, except per equity share data\s*$', t['before'])], 'quarterly USD fact table')
    labels = {'revenue': 'Revenues', 'operating_profit': 'Operating profit', 'net_profit': 'Net profit (after non-controlling interests)', 'operating_margin': 'Operating margin %'}
    values = {}
    for metric, label in labels.items():
        row = one([r for r in table['rows'] if r and r[0] == label], label)
        if len(row) != 6:
            raise ValueError('Fact-table columns changed')
        values[metric] = (decimal(row[4]), decimal(row[1]))
    if not re.search(r'International Financial Reporting Standards \(IFRS\) in US Dollars', results.text) or 'Dollars in millions except equity share data' not in results.text:
        raise ValueError('Statement currency or basis not verified')
    statement = one([t for t in results.tables if t['rows'] and t['rows'][0][:3] ==
                     ['Condensed Consolidated Statement of Comprehensive Income for the', 'Note', 'Three months ended']], 'consolidated quarterly statement')
    full = lambda d: d.strftime('%B') + f' {d.day}, {d.year}'
    if len(statement['rows'][1]) < 4 or statement['rows'][1][:4] != ['', '', full(end), full(prior_year)]:
        raise ValueError('Statement quarter columns changed')
    for metric, label in [('revenue', 'Revenues'), ('operating_profit', 'Operating profit'), ('net_profit', 'Owners of the Company')]:
        rows = statement['rows']
        matches = [r for i, r in enumerate(rows) if r and r[0] == label and
                   (metric != 'net_profit' or i > 0 and rows[i - 1][0] == 'Profit attributable to:')]
        row = one(matches, 'statement ' + label)
        if len(row) != len(rows[1]) or decimal(row[2]) != values[metric][1]:
            raise ValueError('Official documents disagree or statement columns changed')
    return previous.isoformat(), values


def build():
    raw, stamp, digest = fetch(FEED, 'application/json')
    discovery = discover(json.loads(raw), stamp, digest)
    try:
        raw, _, _ = fetch(discovery['url'], 'text/html')
        urls = exhibit_links(raw.decode('utf-8'), discovery)
        sources, documents = [], {}
        for kind, url in urls.items():
            raw, stamp, digest = fetch(url, 'text/html')
            documents[kind] = raw.decode('utf-8')
            sources.append({'id': 'infy-sec-' + kind, 'title': 'Infosys SEC-filed ' + ('fact sheet' if kind == 'facts' else 'consolidated IFRS USD statements'),
                            'url': url, 'sha256': digest, 'publishedOn': discovery['filedOn'], 'fetchedAt': stamp,
                            'kind': kind, 'mediaType': 'text/html', 'fingerprintKind': 'document_bytes', 'retrieval': 'direct', 'pages': 1, 'bytes': len(raw)})
        previous, values = extract(documents['facts'], documents['results'], discovery['periodEnd'])
        comparisons = []
        for metric, (old, current) in values.items():
            def point(value, date):
                return {'value': value, 'unit': '%' if metric == 'operating_margin' else 'USD million', 'basis': 'consolidated', 'standard': 'IFRS',
                        'treatment': 'reported', 'periodEnd': date, 'periodType': 'quarter', 'sourceId': 'infy-sec-facts', 'page': None,
                        'locator': 'Consolidated Statement of Comprehensive Income; three months; USD table; ' + date + ' column'}
            comparisons.append({'metric': metric, 'previous': point(old, previous), 'current': point(current, discovery['periodEnd']),
                                'note': 'Published rounded margin; difference in percentage points.' if metric == 'operating_margin' else 'Current value cross-checked against the separate consolidated IFRS USD statement.'})
        draft = {'schemaVersion': 1, 'symbol': 'INFY', 'title': 'Quarterly results: revenue, operating profit and margin', 'periodEnd': discovery['periodEnd'], 'comparison': 'QoQ',
                 'sources': sources, 'comparisons': comparisons, 'extractorVersion': 'infy-sec-tables-v1',
                 'limitations': ['Sources are Infosys exhibits furnished to the SEC. Source publication dates are SEC filing dates, which can follow the company announcement.',
                                 'Reported USD growth includes currency effects. Prior-quarter values come directly from the fact sheet; no half-year subtraction is used.',
                                 'Only the recent SEC results feed is monitored. Filings can appear after the company release; this is not comprehensive exchange-announcement coverage.',
                                 'Selected reported metrics only; no valuation, forecast or trading recommendation. Original tables and accounting notes require human review.']}
        return {'draft': draft, 'discovery': discovery, 'check': {'symbol': 'INFY', 'checkedAt': min(s['fetchedAt'] for s in sources), 'status': 'success',
                'message': 'Latest SEC results filing discovered; fact sheet and IFRS USD statements retrieved and cross-checked. Human publication review required.', 'fingerprints': [s['sha256'] for s in sources]}}
    except (ValueError, OSError, KeyError) as exc:
        # Preserve discovery even when the newly discovered release is not extractable.
        error = DiscoveryError('Latest SEC results filing found, but its documents could not be retrieved or validated. Review the filing; no older quarter was substituted.', discovery)
        error.__cause__ = exc
        raise error
