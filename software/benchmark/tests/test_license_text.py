from pathlib import Path
import re


def test_installer_license_is_plain_text_with_unchanged_terms():
    app=Path(__file__).resolve().parents[1]
    original=(app/'LICENSE').read_text(encoding='utf-8')
    expected=re.sub(r'\[([^\]]+)\]\([^)]*\)',r'\1',original)
    expected=re.sub(r'^#{1,6} +','',expected,flags=re.M)
    expected=re.sub(r'^> ?','',expected,flags=re.M)
    expected=re.sub(r'<(https?://[^>]+)>',r'\1',expected).replace('*','')
    actual=(app/'LICENSE.txt').read_text(encoding='utf-8')
    assert actual==expected
    assert not re.search(r'^#|\*\*|\]\(#',actual,re.M)
    assert 'LicenseFile={#SourceDir}\\LICENSE.txt' in (app/'packaging/windows-installer.iss').read_text()
