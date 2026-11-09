#!/usr/bin/env python3
"""Prepare an unpublished TCS Q2 FY27 draft from an operator-downloaded workbook.

Pinned layout: fails closed on changed dates, labels, basis or missing values.
Does not fetch URLs, run macros or arbitrary formulas, approve, or publish.
"""
import argparse
import datetime as dt
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import posixpath
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile

from batch import ROOT, LIMIT, clean, source_url

URL = 'https://www.tcs.com/content/dam/tcs/pdf/discover-tcs/investor-relations/tcs-data-sheet/TCS-Data-Sheet-Q2FY27.xlsx'
MIME = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
REL = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
FINANCIAL = 'IFRS-PnL,BS-USD'
OPERATING = 'Operating Metrics'


def download_time(value):
    retrieved = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    current = dt.datetime.now(dt.timezone.utc)
    if retrieved.tzinfo is None or retrieved > current + dt.timedelta(minutes=5) or current - retrieved > dt.timedelta(days=7):
        raise ValueError('Use the actual download timestamp, within the last seven days')
    return retrieved.astimezone(dt.timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


class Workbook:
    def __init__(self, raw):
        if not 0 < len(raw) <= LIMIT:
            raise ValueError('Workbook exceeds the 20 MB limit')
        self.archive = zipfile.ZipFile(io.BytesIO(raw))
        entries = self.archive.infolist()
        if len(entries) > 500 or sum(i.file_size for i in entries) > 40 * 1024 * 1024:
            raise ValueError('Expanded workbook exceeds processing limits')
        if len({i.filename for i in entries}) != len(entries):
            raise ValueError('Duplicate workbook members')
        if any(i.flag_bits & 1 or 'vbaproject' in i.filename.lower() for i in entries):
            raise ValueError('Encrypted or macro workbooks are unsupported')
        if self.archive.testzip():
            raise ValueError('Damaged workbook')
        self.strings = [''.join(e.itertext()) for e in self.xml('xl/sharedStrings.xml').findall('m:si', NS)]
        wb = self.xml('xl/workbook.xml')
        properties = wb.find('m:workbookPr', NS)
        if properties is not None and properties.get('date1904') in ('1', 'true'):
            raise ValueError('Unsupported workbook date system')
        rels = {r.get('Id'): r for r in self.xml('xl/_rels/workbook.xml.rels')}
        self.sheets = {}
        self.sheet_paths = {}
        for sheet in wb.findall('m:sheets/m:sheet', NS):
            name = sheet.get('name')
            if name not in (FINANCIAL, OPERATING):
                continue
            if name in self.sheets:
                raise ValueError('Duplicate worksheet')
            rel = rels[sheet.get(REL)]
            if rel.get('TargetMode') == 'External':
                raise ValueError('External worksheet is unsupported')
            target = rel.get('Target', '')
            path = posixpath.normpath(posixpath.join('xl', target)) if not target.startswith('/') else target.lstrip('/')
            if not path.startswith('xl/worksheets/'):
                raise ValueError('Unexpected worksheet path')
            cells = {}
            for c in self.xml(path).findall('m:sheetData/m:row/m:c', NS):
                ref = c.get('r')
                if ref in cells:
                    raise ValueError('Duplicate cell')
                cells[ref] = c
            self.sheets[name] = cells
            self.sheet_paths[name] = path
        if set(self.sheets) != {FINANCIAL, OPERATING}:
            raise ValueError('Required worksheets are missing')

    def xml(self, path):
        data = self.archive.read(path)
        if b'<!DOCTYPE' in data.upper() or b'<!ENTITY' in data.upper():
            raise ValueError('XML declarations are unsupported')
        return ET.fromstring(data)

    def value(self, sheet, ref):
        c = self.sheets[sheet].get(ref)
        if c is None:
            raise ValueError('Required cell is missing: ' + sheet + '!' + ref)
        if c.find('m:f', NS) is not None:
            return self.calculated(sheet, ref, c)
        if c.get('t') == 'inlineStr':
            return ''.join(c.find('m:is', NS).itertext())
        v = c.find('m:v', NS)
        if v is None or v.text is None:
            return ''
        if c.get('t') == 's':
            index = int(v.text)
            if not 0 <= index < len(self.strings):
                raise ValueError('Invalid shared string reference')
            return self.strings[index]
        if c.get('t') not in (None, 'n'):
            raise ValueError('Unsupported source cell type')
        return v.text

    def calculated(self, sheet, ref, c):
        """Validate only the fixed source's simple formulas and cached results.

        No formula interpreter, external references, or arbitrary expressions.
        All precedents below move strictly toward literal input cells.
        """
        f = c.find('m:f', NS)
        formula = f.text
        if f.get('t') == 'shared' and not formula:
            masters = [(r, x.find('m:f', NS)) for r, x in self.sheets[sheet].items()
                       if x.find('m:f', NS) is not None and x.find('m:f', NS).get('si') == f.get('si') and x.find('m:f', NS).text]
            if len(masters) != 1:
                raise ValueError('Ambiguous shared formula')
            anchor, master = masters[0]
            # The accepted financial formulas use only single-letter columns.
            if not re.fullmatch(r'[A-H]\d+', ref) or not re.fullmatch(r'[A-H]\d+', anchor):
                raise ValueError('Unsupported shared formula location')
            dc, dr = ord(ref[0])-ord(anchor[0]), int(ref[1:])-int(anchor[1:])
            formula = re.sub(r'\b([A-H])(\d+)\b', lambda m: chr(ord(m[1])+dc)+str(int(m[2])+dr), master.text)
        if sheet == OPERATING and ref in ('DD1', 'DE1'):
            target = 'G1' if ref == 'DD1' else 'H1'
            if formula != "'" + FINANCIAL + "'!" + target:
                raise ValueError('Period header formula changed')
            expected = self.value(FINANCIAL, target)
        elif sheet == FINANCIAL and re.fullmatch(r'[GH](19|21|25|29|33)', ref):
            col, row = ref[0], int(ref[1:])
            formulas = {19: f'SUM({col}6:{col}18)', 21: f'{col}4-{col}19', 25: f'{col}21+{col}23', 29: f'{col}25-{col}27', 33: f'{col}29+{col}31'}
            if formula != formulas[row]:
                raise ValueError('Financial source formula changed: ' + ref)
            def number(r):
                value = self.value(sheet, col + str(r))
                if not re.fullmatch(r'-?\d{1,15}(?:\.\d{1,6})?', value):
                    raise ValueError('Invalid formula precedent')
                return Decimal(value)
            if row == 19:
                expected = sum(number(r) for r in range(6, 19))
            else:
                left, right, sign = {21:(4,19,-1),25:(21,23,1),29:(25,27,-1),33:(29,31,1)}[row]
                expected = number(left) + sign * number(right)
        else:
            raise ValueError('Unsupported formula in required evidence: ' + ref)
        cached = c.find('m:v', NS)
        if cached is None or cached.text != str(expected):
            raise ValueError('Cached formula result disagrees with its source inputs: ' + ref)
        return cached.text

    def expect(self, sheet, ref, expected):
        if self.value(sheet, ref) != expected:
            raise ValueError('Source layout, period or basis changed: ' + sheet + '!' + ref)

    def reported_comment(self, sheet, ref):
        path = self.sheet_paths[sheet]
        relpath = posixpath.join(posixpath.dirname(path), '_rels', posixpath.basename(path) + '.rels')
        relationships = [r for r in self.xml(relpath) if r.get('Type', '').endswith('/comments')]
        if len(relationships) != 1 or relationships[0].get('TargetMode') == 'External':
            raise ValueError('A unique original comments part is required')
        target = relationships[0].get('Target', '')
        comments_path = posixpath.normpath(posixpath.join(posixpath.dirname(path), target)) if not target.startswith('/') else target.lstrip('/')
        if not comments_path.startswith('xl/'):
            raise ValueError('Invalid comments path')
        comments = [c for c in self.xml(comments_path).findall('m:commentList/m:comment', NS) if c.get('ref') == ref]
        if len(comments) != 1:
            raise ValueError('A unique reported-value cell comment is required')
        text = ''.join(comments[0].find('m:text', NS).itertext())
        # Treat comment text as data only. A changed/ambiguous comment must be reviewed.
        match = re.fullmatch(r'(?:[^\n]*:\s*\n)?\s*([\d,]+) including Exceptional Items\s*', text)
        if not match or not re.fullmatch(r'\d{1,3}(?:,\d{3})*|\d+', match[1]):
            raise ValueError('Reported-value comment structure changed')
        return match[1].replace(',', '')


def build(raw, retrieved_at):
    retrieved_at = download_time(retrieved_at)
    if retrieved_at[:10] < '2026-10-08':
        raise ValueError('Download predates the release')
    workbook = Workbook(raw)
    for ref, expected in {'A1': 'TCS Consolidated Income Statement as per IFRS - USD Mn', 'G1': '1Q27', 'H1': '2Q27', 'G2': 'EX-ADJ 1Q27', 'G51': '46203', 'H51': '46295', 'A4': 'Revenue', 'A21': 'Operating Income', 'A33': 'Net profit'}.items():
        workbook.expect(FINANCIAL, ref, expected)
    # Q2 must not silently acquire a different adjustment treatment.
    if 'H2' in workbook.sheets[FINANCIAL] and workbook.value(FINANCIAL, 'H2'):
        raise ValueError('Q2 adjustment treatment changed')
    footnote = workbook.value(FINANCIAL, 'A129')
    if not all(t in footnote for t in ('1Q27 excludes exceptional Items', '$70Mn', '$ 53 Mn', 'Cell Comments')):
        raise ValueError('Exceptional-item footnote changed')
    for ref, expected in {'A1': 'TCS OPERATING METRICS', 'DD1': '1Q27', 'DE1': '2Q27', 'A152': 'Total Headcount'}.items():
        workbook.expect(OPERATING, ref, expected)

    def point(sheet, cell, previous=False, adjusted=False, employees=False):
        value = workbook.value(sheet, cell)
        if not re.fullmatch(r'\d{1,15}(?:\.\d{1,6})?', value) or (employees and '.' in value):
            raise ValueError('Missing or non-exact numeric evidence: ' + sheet + '!' + cell)
        return {'value': value, 'unit': 'people' if employees else 'USD million', 'basis': 'consolidated',
                'standard': 'operational' if employees else 'IFRS',
                'treatment': 'excluding exceptional items' if adjusted else 'reported',
                'periodEnd': '2026-06-30' if previous else '2026-09-30',
                'periodType': 'point_in_time' if employees else 'quarter',
                'sourceId': 'tcs-q2-workbook', 'page': None, 'worksheet': sheet, 'cell': cell,
                'locator': sheet + '!' + cell + ('; Q1 basis: G2 and A129; reported alternative in cell comment' if adjusted else '')}

    comparisons = []
    reported = {21: workbook.reported_comment(FINANCIAL, 'G21'), 33: workbook.reported_comment(FINANCIAL, 'G33')}
    for metric, row in [('revenue', 4), ('operating_profit', 21), ('net_profit', 33)]:
        adjusted = metric != 'revenue'
        comparisons.append({'metric': metric, 'previous': point(FINANCIAL, 'G' + str(row), True, adjusted),
                            'current': point(FINANCIAL, 'H' + str(row)),
                            'note': ('Q1 displayed profit excludes exceptional items; Q2 is reported. No growth is calculated across these treatments. The G' + str(row) + ' comment gives Q1 reported profit of USD ' + reported[row] + ' million; reconcile against results before using it.') if adjusted else 'Both quarter values come from the same USD revenue row.'})
    comparisons.append({'metric': 'employees', 'previous': point(OPERATING, 'DD152', True, employees=True),
                        'current': point(OPERATING, 'DE152', employees=True), 'note': 'Quarter-end total headcount, not hiring or net recruitment.'})
    sha = hashlib.sha256(raw).hexdigest()
    source = {'id': 'tcs-q2-workbook', 'title': 'TCS Q2 FY27 financial and operating data sheet', 'url': source_url(URL, 'TCS'),
              'sha256': sha, 'publishedOn': '2026-10-08', 'fetchedAt': retrieved_at,
              'kind': 'facts', 'mediaType': MIME, 'fingerprintKind': 'document_bytes', 'retrieval': 'operator_file', 'pages': 0, 'bytes': len(raw)}
    draft = {'schemaVersion': 1, 'symbol': 'TCS', 'title': 'TCS Q2 FY27 workbook comparison — supporting results required',
             'periodEnd': '2026-09-30', 'comparison': 'QoQ', 'sources': [source], 'comparisons': comparisons,
             'limitations': ['Operator-downloaded workbook. Automatic source access remains unverified; this import is not a successful scheduled run.',
                            'Only one supporting data sheet is attached. Obtain the official consolidated results and reconcile the figures before publication.',
                            'Q1 displayed profits exclude exceptional items. The workbook contains reported alternatives in cell comments; these have not been substituted into this draft.',
                            'The Q1 net-profit footnote states a USD 53 million adjustment. The displayed value (' + workbook.value(FINANCIAL, 'G33') + ') and G33 cell-comment value (' + reported[33] + ') differ by USD ' + str(Decimal(workbook.value(FINANCIAL, 'G33')) - Decimal(reported[33])) + ' million. Reconcile these disclosures against the official results.',
                            'Contract value is not extracted from this workbook. Missing disclosures are not inferred.'],
             'extractorVersion': 'tcs-operator-workbook-q2fy27-v1'}
    return {'draft': draft, 'check': {'symbol': 'TCS', 'checkedAt': source['fetchedAt'], 'status': 'success',
            'message': 'Operator-downloaded workbook validated; automatic retrieval remains unverified. Supporting results and accounting reconciliation are required.', 'fingerprints': [sha]}}


def statement_values(page, quarter):
    """Read the current quarter column only; never derive Q1 from half-year totals."""
    compact = clean(page)
    if 'TATA CONSULTANCY SERVICES LIMITED' not in compact or 'Unaudited consolidated interim statement of profit or loss and other comprehensive income' not in compact or 'In million of USD, except shares' not in compact:
        raise ValueError('Results identity, accounting scope or currency changed')
    header = compact.split('Revenue')[0]
    expected = ('Three months Three months Six months Six months Note ended ended ended ended September 30, September 30, September 30, September 30, 2026 2025 2026 2025' if quarter == 2 else 'Note Three months Three months ended ended June 30, 2026 June 30, 2025')
    if expected not in header:
        raise ValueError('Results quarter columns changed')
    count = 4 if quarter == 2 else 2
    numbers = r'([\d,]+)' + r'\s+[\d,]+' * (count - 1)
    def row(label, note='', section=None):
        body = page
        if section:
            if body.count(section) != 1:
                raise ValueError('Ambiguous statement section')
            body = body.split(section)[1].split('Other comprehensive income for the period attributable to:')[0]
        matches = re.findall(r'^\s*' + re.escape(label) + (r'\s+' + note if note else '') + r'\s+' + numbers + r'\s*$', body, re.M)
        if len(matches) != 1:
            raise ValueError('Ambiguous or missing results row: ' + label)
        return matches[0].replace(',', '')
    values = {'revenue': row('Revenue', '12'), 'operating_profit': row('Operating profit'),
              'net_profit': row('Shareholders of the Company', section='Profit for the period attributable to:')}
    owners = int(values['net_profit'])
    noncontrolling = int(row('Non-controlling interests', section='Profit for the period attributable to:'))
    if owners + noncontrolling != int(row('Profit for the period')):
        raise ValueError('Owners and non-controlling profit do not reconcile')
    return values


def read_results(path, quarter, retrieved_at):
    if path.stat().st_size > LIMIT:
        raise ValueError('PDF exceeds the 20 MB limit')
    raw = path.read_bytes()
    if not raw.startswith(b'%PDF-'):
        raise ValueError('Invalid PDF signature')
    info = subprocess.run(['pdfinfo', str(path)], capture_output=True, check=True, timeout=15).stdout.decode()
    count = re.search(r'^Pages:\s+(\d+)', info, re.M)
    if not count or int(count[1]) != (53 if quarter == 1 else 57):
        raise ValueError('Results PDF page count changed')
    extracted = subprocess.run(['pdftotext', '-layout', str(path), '-'], capture_output=True, check=True, timeout=30).stdout
    if len(extracted) > LIMIT:
        raise ValueError('PDF text exceeds extraction limit')
    pages = extracted.decode().split('\f')
    if 'prepared in accordance with International Financial Reporting Standards' not in clean(pages[5]) or 'IAS 34' not in pages[5]:
        raise ValueError('Original IFRS accounting-basis note changed')
    values = statement_values(pages[1], quarter)
    if quarter == 2 and not all(s in clean(pages[50]) for s in ('US $70 million towards exemplary damages', 'ended June 30, 2026', 'Settlement of legal claim')):
        raise ValueError('Q1 exceptional-item supporting note changed')
    source = {'id': f'tcs-q{quarter}-ifrs-file', 'title': f'TCS Q{quarter} FY27 condensed consolidated IFRS results (USD)',
              'url': source_url(f'https://www.tcs.com/content/dam/tcs/investor-relations/financial-statements/2026-27/q{quarter}/IFRS/Condensed%20Consolidated,%20Unaudited%20-%20USD.pdf', 'TCS'),
              'sha256': hashlib.sha256(raw).hexdigest(), 'publishedOn': '2026-07-09' if quarter == 1 else '2026-10-08',
              'fetchedAt': download_time(retrieved_at), 'kind': 'results', 'mediaType': 'application/pdf',
              'fingerprintKind': 'document_bytes', 'retrieval': 'operator_file', 'pages': int(count[1]), 'bytes': len(raw)}
    if source['fetchedAt'][:10] < source['publishedOn']:
        raise ValueError('PDF download predates publication')
    return source, values, raw


def reconcile_results(bundle, raw_workbook, statements):
    workbook = Workbook(raw_workbook)
    draft = bundle['draft']
    for quarter, (source, values, _) in enumerate(statements, 1):
        side = 'previous' if quarter == 1 else 'current'
        for comparison in draft['comparisons']:
            metric = comparison['metric']
            if metric not in values:
                continue
            point = comparison[side]
            expected = workbook.reported_comment(FINANCIAL, point['cell']) if quarter == 1 and metric != 'revenue' else point['value']
            if values[metric] != expected:
                raise ValueError('Workbook and original reported results disagree: ' + side + ' ' + metric)
            point.update(value=values[metric], sourceId=source['id'], page=2, treatment='reported',
                         locator='Page 2; first three-month column; ' + ('Profit for the period attributable to shareholders of the Company' if metric == 'net_profit' else metric.replace('_', ' ')))
            point.pop('worksheet');point.pop('cell')
            comparison['note'] = ('Reported IFRS values from each original quarter statement, cross-checked against the workbook. Q1 includes the legal-claim expense; this is not adjusted or underlying growth.' if metric != 'revenue' else 'Quarterly USD revenue from both original IFRS statements; cross-checked against the workbook.')
        draft['sources'].append(source)
    draft['title'] = 'TCS Q2 FY27 — reported results reconciled to original statements'
    draft['extractorVersion'] = 'tcs-operator-files-q2fy27-v2'
    draft['limitations'] = [
        'These three original files were downloaded through the browser. Automatic source access remains unverified; this is not a successful scheduled run.',
        'Profit comparisons now use reported values from page 2 of each original IFRS statement. Q1 includes the USD 70 million legal-claim expense described on page 51 of the Q2 results. Reported growth is not underlying or adjusted growth.',
        'The workbook Q1 net-profit adjustment footnote says USD 53 million; its displayed adjusted value and reported cell comment differ by USD 52 million. No explanation is inferred. This draft uses the original statement’s reported profit, corroborated by the cell comment, and does not use that adjustment.',
        'Net profit means profit attributable to shareholders, excluding non-controlling interests. Quarter values are read directly, never derived from half-year totals.',
        'Headcount comes from the workbook only. Contract value is omitted because it was not extracted from these sources.',
        'This is a proposed replacement for the existing TCS publication. Publishing it would create a new version; the earlier version and personal decisions remain in history.'
    ]
    # The completed document set is only available after the final download.
    # Reuse of the earlier workbook check time would collide with its separate check.
    bundle['check'].update(checkedAt=max(s['fetchedAt'] for s in draft['sources']),
                           message='Browser-downloaded original results and workbook reconciled; automatic source access remains unverified. New human review is required before publication.',
                           fingerprints=[s['sha256'] for s in draft['sources']])
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', type=Path)
    parser.add_argument('--retrieved-at', required=True, help='Actual browser download time with timezone; never the replay time')
    parser.add_argument('--q1-results', type=Path)
    parser.add_argument('--q1-retrieved-at')
    parser.add_argument('--q2-results', type=Path)
    parser.add_argument('--q2-retrieved-at')
    parser.add_argument('--output-dir', type=Path, help='Private output directory for this import')
    args = parser.parse_args()
    if args.file.stat().st_size > LIMIT:
        raise ValueError('Workbook exceeds the 20 MB limit')
    raw = args.file.read_bytes()
    bundle = build(raw, args.retrieved_at)
    inputs = [args.q1_results, args.q1_retrieved_at, args.q2_results, args.q2_retrieved_at]
    if any(inputs) and not all(inputs):
        raise ValueError('Provide both original quarter PDFs and their actual download timestamps')
    statements = []
    if all(inputs):
        statements = [read_results(args.q1_results, 1, args.q1_retrieved_at), read_results(args.q2_results, 2, args.q2_retrieved_at)]
        bundle = reconcile_results(bundle, raw, statements)
    sha = bundle['draft']['sources'][0]['sha256']
    cache = ROOT / '.local/research/cache'
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / (sha + '.xlsx')
    if archive.exists() and archive.read_bytes() != raw:
        raise ValueError('Archive fingerprint conflict')
    archive.write_bytes(raw)
    for source, _, pdf in statements:
        (cache / (source['sha256'] + '.pdf')).write_bytes(pdf)
    output = args.output_dir or ROOT / '.local/research/manual'
    output.mkdir(parents=True, exist_ok=True)
    path = output / ('TCS-reconciled.json' if statements else 'TCS-workbook.json')
    path.write_text(json.dumps(bundle, indent=2) + '\n')
    print('Unpublished draft prepared: ' + str(path))
    print('Human review required before publication.' if statements else 'Publication blocked pending supporting results and comparable profit evidence.')


if __name__ == '__main__':
    main()
