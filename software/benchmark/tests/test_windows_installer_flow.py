"""Compile and execute tiny real installers without touching an installed app.

Opt in on Windows with JINGWEI_ISCC pointing to Inno Setup 6.7.3's ISCC.exe.
Interactive directory/browse checks additionally use the production wizard.
"""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from threading import Thread

import pytest
from jingwei_benchmark import __version__

APP = Path(__file__).resolve().parents[1]
ISCC = os.environ.get('JINGWEI_ISCC')
pytestmark = pytest.mark.skipif(sys.platform != 'win32' or not ISCC,
                                reason='opt-in real Windows installer integration')


def run(command, **kwargs):
    arguments = [str(arg) for arg in command]
    deadline = time.monotonic() + 120
    while True:
        try:
            return subprocess.run(arguments, capture_output=True,
                                  text=True, encoding='utf-8', errors='replace',
                                  timeout=90, **kwargs)
        except OSError as error:
            # App Control can temporarily deny a just-built unsigned fixture.
            # Retry the same unchanged file/command only; never alter policy.
            if error.winerror != 4551 or time.monotonic() >= deadline:
                raise
            time.sleep(5)


@pytest.fixture(scope='module')
def payload(tmp_path_factory):
    folder = tmp_path_factory.mktemp('installer-fixture')
    source = folder / 'source'
    source.mkdir()
    (source / '_internal').mkdir()
    (source / '_internal/fixture.txt').write_text('installed exactly once', encoding='utf-8')
    for name in ['jingwei.exe', 'jingwei-cli.exe']:
        (source / name).write_bytes(b'installer test fixture, never launched')
    shutil.copyfile(APP / 'LICENSE.txt', source / 'LICENSE.txt')
    # The compiler only embeds this existing icon; no application code is run.
    icon = Path(ISCC).parent / 'SetupClassicIcon.ico'
    shutil.copyfile(icon, source / 'jingwei.ico')
    script = APP / 'packaging/windows-installer.iss'
    for label, failure in [('ok', False), ('failed', True)]:
        target_script = script
        if failure:
            target_script = folder / 'failure.iss'
            text = script.read_text(encoding='utf-8')
            # Resolve the original script's relative translation path.
            text = text.replace('languages\\ChineseSimplified.isl',
                                str(APP / 'packaging/languages/ChineseSimplified.isl'))
            text += '\nfunction GetCustomSetupExitCode: Integer;\nbegin\n  Result := 23;\nend;\n'
            target_script.write_text(text, encoding='utf-8')
        result = run([ISCC, f'/DSourceDir={source}', f'/DReleaseDir={folder / label}',
                      f'/DAppVersion={__version__}', target_script])
        assert result.returncode == 0, result.stdout + result.stderr
    return folder


@pytest.fixture
def server(payload):
    requests = []
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            super().do_GET()
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, directory=str(payload)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def bootstrap(payload, tmp_path, url, label='ok'):
    result = run([sys.executable, APP / 'packaging/build_bootstrap.py', '--iscc', ISCC,
                  '--payload', payload / label, '--icon', payload / 'source/jingwei.ico',
                  '--output', tmp_path, '--base-url', f'{url}/{label}', '--test-server'])
    assert result.returncode == 0, result.stdout + result.stderr
    return tmp_path / 'jingwei-Windows-Setup.exe'


def install(exe, folder, cache):
    command = [exe, '/JINGWEI-VALIDATION=1', '/VERYSILENT', '/SUPPRESSMSGBOXES',
                '/NORESTART', '/NOICONS', f'/DIR={folder}', f'/CACHE={cache}',
                f'/LOG={folder.parent / "bootstrap.log"}',
                f'/PAYLOADLOG={folder.parent / "payload.log"}']
    # Fresh unsigned test fixtures can be temporarily denied by this machine's
    # App Control while their reputation is checked. Retry only the exact same
    # files/command on this explicitly identified OS error, without changing
    # any security setting or treating other installation errors as transient.
    deadline = time.monotonic() + 120
    while True:
        result = run(command)
        log = (folder.parent / 'bootstrap.log').read_text(encoding='utf-8-sig')
        blocked = '应用程序控制策略' in log or 'application control policy' in log.lower()
        if not blocked or time.monotonic() >= deadline:
            return result
        time.sleep(5)


def test_single_install_into_chinese_and_space_path(payload, server, tmp_path):
    url, requests = server
    exe = bootstrap(payload, tmp_path / 'build', url)
    target = tmp_path / '自定义 安装目录/jingwei'
    target.parent.mkdir()
    cache = tmp_path / 'cache'
    result = install(exe, target, cache)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (target / '_internal/fixture.txt').read_text() == 'installed exactly once'
    assert not list(target.glob('unins*'))
    assert not list(cache.glob('*'))
    log = (target.parent / 'bootstrap.log').read_text(encoding='utf-8-sig')
    assert log.count('Installing payload once into:') == 1
    log = (target.parent / 'payload.log').read_text(encoding='utf-8-sig')
    assert '/VERYSILENT /SUPPRESSMSGBOXES' in log
    assert str(target) in log
    assert len(requests) == len(list((payload / 'ok').iterdir()))


def test_cached_payload_repairs_only_corrupt_slice(payload, server, tmp_path):
    url, requests = server
    exe = bootstrap(payload, tmp_path / 'build', url)
    cache = tmp_path / 'cache'
    cache.mkdir()
    for source in (payload / 'ok').iterdir():
        shutil.copyfile(source, cache / source.name)
    damaged = next(cache.glob('*.bin'))
    damaged.write_bytes(b'bad download')
    result = install(exe, tmp_path / 'app', cache)
    assert result.returncode == 0
    assert requests == ['/ok/' + damaged.name]
    assert not list(cache.glob('*'))


def test_child_failure_is_not_reported_as_success_and_keeps_retry_cache(payload, server, tmp_path):
    url, requests = server
    exe = bootstrap(payload, tmp_path / 'build', url, label='failed')
    cache = tmp_path / 'cache'
    result = install(exe, tmp_path / 'app', cache)
    assert result.returncode != 0
    assert len(list(cache.glob('*'))) == len(requests) > 0
    log = (tmp_path / 'bootstrap.log').read_text(encoding='utf-8-sig')
    assert 'code 23' in log or '代码 23' in log


def test_failed_download_does_not_start_installation(payload, server, tmp_path):
    url, requests = server
    exe = bootstrap(payload, tmp_path / 'build', url)
    # Stop serving this URL, while keeping the compiled, pinned manifest.
    # A query-free alternate path gets a deterministic HTTP 404.
    bin_file = next((payload / 'ok').glob('*.bin'))
    hidden = bin_file.with_suffix('.unavailable')
    bin_file.rename(hidden)
    try:
        result = install(exe, tmp_path / 'app', tmp_path / 'cache')
    finally:
        hidden.rename(bin_file)
    assert result.returncode != 0
    assert not (tmp_path / 'payload.log').exists()
    assert not (tmp_path / 'app/_internal/fixture.txt').exists()


def test_production_manifest_rejects_non_github_urls(payload, tmp_path):
    result = run([sys.executable, APP / 'packaging/build_bootstrap.py', '--iscc', ISCC,
                  '--payload', payload / 'ok', '--icon', payload / 'source/jingwei.ico',
                  '--output', tmp_path, '--base-url', 'http://127.0.0.1/untrusted'])
    assert result.returncode != 0
    assert 'production payloads must use HTTPS on github.com' in result.stderr
    assert not (tmp_path / 'jingwei-Windows-Setup.exe').exists()


@pytest.mark.parametrize('setup_language,expected', [('english', 'english'), ('chinesesimp', 'chinesesimp')])
def test_complete_offline_exe_installs_without_adjacent_payload(payload, tmp_path, setup_language, expected):
    output = tmp_path / 'offline-build'
    result = run([ISCC, '/DSingleFileInstaller', f'/DSourceDir={payload / "source"}',
                  f'/DReleaseDir={output}', f'/DAppVersion={__version__}',
                  APP / 'packaging/windows-installer.iss'])
    assert result.returncode == 0, result.stdout + result.stderr
    exe = output / f'jingwei-{__version__}-windows-x86_64-offline.exe'
    assert exe.is_file() and list(output.iterdir()) == [exe]
    # Run the EXE alone from another directory, without the build source/BINs.
    download = tmp_path / '下载 文件夹'
    download.mkdir()
    standalone = download / exe.name
    shutil.copyfile(exe, standalone)
    target = tmp_path / '用户 自定义安装位置/jingwei'
    temp = tmp_path / 'installer-temp'
    temp.mkdir()
    result = run([standalone, '/JINGWEI-VALIDATION=1', '/VERYSILENT',
                  '/SUPPRESSMSGBOXES', '/NORESTART', '/NOICONS',
                  '/NOCLOSEAPPLICATIONS', f'/LANG={setup_language}', f'/DIR={target}', f'/LOG={tmp_path / "offline.log"}'],
                 env={**os.environ, 'TEMP': str(temp), 'TMP': str(temp)}, cwd=download)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (target / '_internal/fixture.txt').read_text() == 'installed exactly once'
    assert (target / 'LICENSE.txt').read_bytes() == (APP / 'LICENSE.txt').read_bytes()
    assert (target / 'installation-language.txt').read_text().splitlines()[0] == expected
    assert not list(target.glob('unins*'))
    assert not (target / '.jingwei-installer-cache').exists()
    assert not list(download.glob('*.bin'))
    log = (tmp_path / 'offline.log').read_text(encoding='utf-8-sig')
    assert str(target) in log and 'Downloading' not in log


def test_optional_online_payload_uses_selected_destination_volume(payload, server, tmp_path):
    url, requests = server
    exe = bootstrap(payload, tmp_path / 'build', url)
    target = tmp_path / 'selected drive 用户/jingwei'
    target.parent.mkdir()
    result = run([exe, '/JINGWEI-VALIDATION=1', '/VERYSILENT', '/SUPPRESSMSGBOXES',
                  '/NORESTART', '/NOICONS', f'/DIR={target}', f'/LOG={tmp_path / "online.log"}'])
    assert result.returncode == 0, result.stdout + result.stderr
    assert (target / '_internal/fixture.txt').is_file()
    assert not (target / '.jingwei-installer-cache').exists()
    assert len(requests) > 0
    log = (tmp_path / 'online.log').read_text(encoding='utf-8-sig')
    assert str(target / '.jingwei-installer-cache') in log


def test_builder_defaults_to_complete_offline_exe(payload, tmp_path):
    source = tmp_path / 'source'
    shutil.copytree(payload / 'source', source)
    (source / 'build-info.json').write_text(json.dumps({'app_version': __version__}))
    output = tmp_path / 'release'
    result = run([sys.executable, APP / 'packaging/build_installer.py', '--iscc', ISCC,
                  '--source', source, '--output', output])
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output / 'windows-offline-installer.json').read_text())
    exe = output / manifest['installer']
    assert exe.is_file() and exe.name.endswith('-offline.exe')
    assert not list(output.glob('*.bin'))
    assert manifest['offline'] and not manifest['payload_downloads']
    assert manifest['destination_editable'] and manifest['destination_browse']
    assert manifest['sha256'] == hashlib.sha256(exe.read_bytes()).hexdigest()
    assert manifest['size'] == exe.stat().st_size
    assert manifest['installed_size'] == sum(p.stat().st_size for p in source.rglob('*') if p.is_file())
