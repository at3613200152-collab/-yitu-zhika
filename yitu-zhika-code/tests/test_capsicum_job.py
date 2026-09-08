"""Offline regression tests: a real HTTP connection breaks mid-download."""
import hashlib
import io
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
PAYLOAD = bytes(range(256)) * 2048


class BrokenOnce(BaseHTTPRequestHandler):
    calls = []
    ignore_range = False

    def log_message(self, *args):
        pass

    def do_GET(self):
        start = int(self.headers.get('Range', 'bytes=0-').split('=')[1].split('-')[0])
        type(self).calls.append(start)
        if self.ignore_range:
            start = 0
        self.send_response(206 if start else 200)
        self.send_header('Content-Length', str(len(PAYLOAD) - start))
        if start:
            self.send_header('Content-Range', f'bytes {start}-{len(PAYLOAD)-1}/{len(PAYLOAD)}')
        self.end_headers()
        if len(self.calls) == 1 and not self.ignore_range:
            self.wfile.write(PAYLOAD[:131072])
            self.wfile.flush()
            # Let the client commit some bytes before injecting a TCP reset.
            time.sleep(0.1)
            fmt = 'hh' if os.name == 'nt' else 'ii'
            self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack(fmt, 1, 0))
            self.connection.close()
        else:
            try:
                self.wfile.write(PAYLOAD[start:])
            except (ConnectionError, OSError):
                pass


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='capsicum-test-')
        self.root = Path(self.tmp.name)
        BrokenOnce.calls = []
        BrokenOnce.ignore_range = False
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), BrokenOnce)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}/file'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.tmp.cleanup()

    def test_connection_reset_resumes_without_corruption(self):
        dest = self.root / 'download.bin'
        if os.environ.get('CAPSICUM_LEGACY'):
            r = subprocess.run(['curl.exe', '-L', '--fail', '--retry', '2',
                                '--retry-delay', '1', '--continue-at', '-',
                                '--max-time', '5', '-o', str(dest), self.url],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
        else:
            from scripts.capsicum_job import resume_download
            resume_download(self.url, dest, len(PAYLOAD), lambda **kw: None,
                            min_free_bytes=0, retry_delay=0.01, max_seconds=10)
        self.assertEqual(dest.read_bytes(), PAYLOAD)
        self.assertGreater(len(BrokenOnce.calls), 1)
        self.assertGreater(BrokenOnce.calls[1], 0)

    def test_server_ignoring_range_does_not_append_wrong_data(self):
        from scripts.capsicum_job import resume_download
        dest = self.root / 'download.bin'
        dest.write_bytes(PAYLOAD[:1024])
        BrokenOnce.ignore_range = True
        with self.assertRaises(ValueError):
            resume_download(self.url, dest, len(PAYLOAD), lambda **kw: None,
                            min_free_bytes=0, retry_delay=0.01, max_seconds=10)
        self.assertEqual(dest.read_bytes(), PAYLOAD[:1024])

    def test_checksum_mismatch_blocks_use(self):
        from scripts.capsicum_job import verify_archive
        p = self.root / 'bad.bin'
        p.write_bytes(b'bad')
        with self.assertRaises(ValueError):
            verify_archive(p, 3, hashlib.md5(b'yes').hexdigest())

    def test_tar_traversal_is_rejected_before_extract(self):
        from scripts.capsicum_job import extract_archive
        p = self.root / 'bad.tar.gz'
        with tarfile.open(p, 'w:gz') as t:
            m = tarfile.TarInfo('../escape.txt')
            m.size = 1
            t.addfile(m, io.BytesIO(b'x'))
        with self.assertRaises(ValueError):
            extract_archive(p, self.root / 'extract', lambda **kw: None, min_free_bytes=0)
        self.assertFalse((self.root / 'escape.txt').exists())


if __name__ == '__main__':
    unittest.main()
