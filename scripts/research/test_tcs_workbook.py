import datetime as dt
import hashlib
import io
import unittest
import xml.etree.ElementTree as ET
import zipfile
from tcs_workbook import build, statement_values, reconcile_results, FINANCIAL, OPERATING, NS

M = NS['m']


def fixture(changes=None, comment='20 including Exceptional Items'):
    """Synthetic layout, never submitted as company research."""
    cells = {'A1':'TCS Consolidated Income Statement as per IFRS - USD Mn', 'G1':'1Q27', 'H1':'2Q27',
             'G2':'EX-ADJ 1Q27', 'G51':46203, 'H51':46295, 'A4':'Revenue', 'A21':'Operating Income', 'A33':'Net profit',
             'A129':'1Q27 excludes exceptional Items; Impact on Operating Margin $70Mn; Net Margin $ 53 Mn. Cell Comments',
             'G4':100, 'H4':120, 'G21':30, 'H21':40, 'G33':21, 'H33':28}
    cells.update(changes or {})
    def sheet(values):
        root = ET.Element('worksheet', xmlns=M)
        data = ET.SubElement(root, 'sheetData')
        for ref, value in values.items():
            c = ET.SubElement(ET.SubElement(data, 'row'), 'c', r=ref)
            if isinstance(value, str):
                c.set('t', 'inlineStr')
                ET.SubElement(ET.SubElement(c, 'is'), 't').text = value
            elif isinstance(value, tuple):
                ET.SubElement(c, 'f').text = value[0]
                ET.SubElement(c, 'v').text = str(value[1])
            else:
                ET.SubElement(c, 'v').text = str(value)
        return ET.tostring(root)
    comments = ET.Element('comments', xmlns=M)
    listing = ET.SubElement(comments, 'commentList')
    for ref, value in [('G21','25 including Exceptional Items'), ('G33',comment)]:
        ET.SubElement(ET.SubElement(ET.SubElement(listing, 'comment', ref=ref), 'text'), 't').text = value
    parts = {
        'xl/sharedStrings.xml': f'<sst xmlns="{M}"/>',
        'xl/workbook.xml': f'<workbook xmlns="{M}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="{FINANCIAL}" r:id="f"/><sheet name="{OPERATING}" r:id="o"/></sheets></workbook>',
        'xl/_rels/workbook.xml.rels': '<Relationships><Relationship Id="f" Target="worksheets/sheet1.xml"/><Relationship Id="o" Target="worksheets/sheet2.xml"/></Relationships>',
        'xl/worksheets/sheet1.xml': sheet(cells),
        'xl/worksheets/sheet2.xml': sheet({'A1':'TCS OPERATING METRICS','DD1':'1Q27','DE1':'2Q27','A152':'Total Headcount','DD152':1000,'DE152':1010}),
        'xl/worksheets/_rels/sheet1.xml.rels': '<Relationships><Relationship Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments" Target="../comments1.xml"/></Relationships>',
        'xl/comments1.xml': ET.tostring(comments),
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w') as archive:
        for name, content in parts.items():
            archive.writestr(name, content)
    return output.getvalue()


class TcsWorkbookTests(unittest.TestCase):
    def test_statement_column_owner_profit_and_cross_document_reconciliation(self):
        heading = 'TATA CONSULTANCY SERVICES LIMITED\nUnaudited consolidated interim statement of profit or loss and other comprehensive income\n'
        header = 'Note Three months Three months ended ended June 30, 2026 June 30, 2025\n(In million of USD, except shares and per share data)\n'
        rows = 'Revenue 12 100 90\nOperating profit 25 23\nProfit for the period 22 19\nProfit for the period attributable to:\nShareholders of the Company 20 18\nNon-controlling interests 2 1\nOther comprehensive income for the period attributable to:\n'
        page = heading + header + rows
        self.assertEqual(statement_values(page,1),{'revenue':'100','operating_profit':'25','net_profit':'20'})
        for changed in (page.replace('June 30, 2026','March 31, 2026'), page.replace('million of USD','million of INR'), page.replace('period 22 19','period 23 19'), page + 'Revenue 12 999 900\n'):
            with self.assertRaises(ValueError): statement_values(changed,1)
        q2_header = 'Three months Three months Six months Six months Note ended ended ended ended September 30, September 30, September 30, September 30, 2026 2025 2026 2025\n(In million of USD, except shares and per share data)\n'
        q2_rows = 'Revenue 12 120 99 220 189\nOperating profit 40 30 65 53\nProfit for the period 31 25 53 44\nProfit for the period attributable to:\nShareholders of the Company 28 23 48 41\nNon-controlling interests 3 2 5 3\nOther comprehensive income for the period attributable to:\n'
        q2 = statement_values(heading + q2_header + q2_rows,2)
        self.assertEqual(q2['net_profit'],'28')  # Neither total profit nor six-month owners' profit.
        raw=fixture();bundle=build(raw,dt.datetime.now(dt.timezone.utc).isoformat())
        source={**bundle['draft']['sources'][0],'mediaType':'application/pdf','kind':'results','pages':53}
        statements=[({**source,'id':'q1'},statement_values(page,1),b''),({**source,'id':'q2'},q2,b'')]
        result=reconcile_results(bundle,raw,statements)
        self.assertEqual(result['draft']['comparisons'][1]['previous']['value'],'25')
        self.assertEqual(result['draft']['comparisons'][1]['previous']['treatment'],'reported')
        self.assertNotIn('cell',result['draft']['comparisons'][1]['previous'])
        self.assertEqual(result['draft']['comparisons'][3]['current']['cell'],'DE152')
        statements[1][1]['revenue']='999'
        with self.assertRaisesRegex(ValueError,'disagree'):
            reconcile_results(build(raw,dt.datetime.now(dt.timezone.utc).isoformat()),raw,statements)

    def test_provenance_adjustments_and_replay(self):
        raw = fixture()
        stamp = (dt.datetime.now(dt.timezone.utc)-dt.timedelta(hours=1)).isoformat()
        one = build(raw, stamp)
        self.assertEqual(one, build(raw, stamp))
        source = one['draft']['sources'][0]
        self.assertEqual(source['sha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(source['retrieval'], 'operator_file')
        self.assertEqual(source['pages'], 0)
        self.assertEqual(source['fetchedAt'], one['check']['checkedAt'])
        revenue, profit, net, employees = one['draft']['comparisons']
        self.assertEqual(revenue['current']['cell'], 'H4')
        self.assertEqual(profit['previous']['treatment'], 'excluding exceptional items')
        self.assertEqual(profit['current']['treatment'], 'reported')
        self.assertEqual(net['previous']['value'], '21')
        self.assertEqual(employees['current']['periodType'], 'point_in_time')

    def test_changed_dates_basis_labels_or_values_fail_closed(self):
        for change in ({'G51':46204}, {'G1':'FY27'}, {'H2':'EX-ADJ'}, {'A4':'Standalone Revenue'}, {'H4':'unknown'}, {'H4':('WEBSERVICE("https://example.test")',120)}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                build(fixture(change), dt.datetime.now(dt.timezone.utc).isoformat())

    def test_corrupt_file_ambiguous_comments_and_stale_download(self):
        with self.assertRaises(zipfile.BadZipFile):
            build(fixture()[:100], dt.datetime.now(dt.timezone.utc).isoformat())
        with self.assertRaisesRegex(ValueError, 'comment structure'):
            build(fixture(comment='20 or 21 including Exceptional Items'), dt.datetime.now(dt.timezone.utc).isoformat())
        for age in (8, -1):
            with self.assertRaisesRegex(ValueError, 'download timestamp'):
                build(fixture(), (dt.datetime.now(dt.timezone.utc)-dt.timedelta(days=age)).isoformat())

    def test_fixed_formulas_must_reconcile_to_literals(self):
        change = {'G29':26, 'G31':-5, 'G33':('G29+G31',21)}
        self.assertEqual(build(fixture(change), dt.datetime.now(dt.timezone.utc).isoformat())['draft']['comparisons'][2]['previous']['value'], '21')
        change['G33'] = ('G29+G31',22)
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            build(fixture(change), dt.datetime.now(dt.timezone.utc).isoformat())
        change['G33'] = ('G33+G31',21)
        with self.assertRaisesRegex(ValueError, 'formula changed'):
            build(fixture(change), dt.datetime.now(dt.timezone.utc).isoformat())


if __name__ == '__main__':
    unittest.main()
