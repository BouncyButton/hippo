"""Local HTTP regression for Drive's 308 Resume Incomplete response."""
import hashlib
import http.server
import json
import tempfile
import threading
import types
import unittest
from unittest.mock import patch
from pathlib import Path

try:
    import httplib2
    from googleapiclient.http import HttpRequest, MediaFileUpload, build_http
except ImportError:
    httplib2 = None


@unittest.skipIf(httplib2 is None, 'Google HTTP libraries are optional locally')
class ResumableTransportTest(unittest.TestCase):
    def test_real_connection_drop_reconciles_old_ledger_and_retries_same_file(self):
        from scripts.colab_msd_cloud import store
        current = [b'{"charged_attempts": 41}']
        attempts = []
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def metadata(self):
                data=json.dumps(dict(id='ledger',parents=['store'],size=str(len(current[0])),
                                     md5Checksum=hashlib.md5(current[0]).hexdigest())).encode()
                self.send_response(200);self.send_header('Content-Length',str(len(data)))
                self.end_headers();self.wfile.write(data)
            def do_GET(self): self.metadata()
            def do_PUT(self):
                body=self.rfile.read(int(self.headers['Content-Length']))
                attempts.append(body)
                if len(attempts)<=2:
                    # httplib2 internally retries one stale connection. Exhaust
                    # that retry too, so our reconciliation path is exercised.
                    self.close_connection=True  # Socket drops before commit/response.
                    return
                current[0]=body;self.metadata()
        server=http.server.HTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        transport=build_http();transport.proxy_info=None;transport.timeout=2
        url=f'http://127.0.0.1:{server.server_port}/ledger'
        class Files:
            def update(self,fileId,media_body,**kwargs):
                return HttpRequest(transport,lambda response,body:json.loads(body),url,
                                   method='PUT',body=Path(media_body._filename).read_bytes())
            def get(self,**kwargs):
                return HttpRequest(transport,lambda response,body:json.loads(body),url,method='GET')
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'charge.json';path.write_bytes(b'{"charged_attempts": 42}')
                cloud=store.Cloud.__new__(store.Cloud);cloud.folder='store';cloud.owned={'ledger'}
                cloud.service=lambda:types.SimpleNamespace(files=lambda:Files())
                with patch.object(store,'retry_wait') as wait:
                    receipt=cloud.upload(path,file_id='ledger')
                self.assertEqual(attempts,[path.read_bytes()]*3)
                self.assertEqual(current[0],path.read_bytes());self.assertEqual(receipt['id'],'ledger')
                self.assertEqual(wait.call_count,1)
        finally:
            transport.close();server.shutdown();thread.join();server.server_close()

    def test_multi_chunk_308_without_redirect_location(self):
        payload = b'checkpoint-state\n' * 180000
        received = bytearray()
        responses = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get('Content-Length', 0)))
                self.send_response(200)
                self.send_header('Location', f'http://127.0.0.1:{self.server.server_port}/session')
                self.end_headers()

            def do_PUT(self):
                received.extend(self.rfile.read(int(self.headers['Content-Length'])))
                done = len(received) == len(payload)
                responses.append(200 if done else 308)
                self.send_response(responses[-1])
                if done:
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'size': len(received),
                        'md5Checksum': hashlib.md5(received).hexdigest()}).encode())
                else:
                    self.send_header('Range', f'bytes=0-{len(received)-1}')
                    self.end_headers()  # Deliberately no redirect Location.

        server = http.server.HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'checkpoint.bin'
                path.write_bytes(payload)
                transport = build_http()
                transport.timeout = 120
                transport.proxy_info = None
                self.assertNotIn(308, transport.redirect_codes)
                self.assertIn(308, httplib2.Http().redirect_codes)
                request = HttpRequest(transport, lambda response, body: json.loads(body),
                    f'http://127.0.0.1:{server.server_port}/upload', method='POST', body='{}',
                    headers={'content-type': 'application/json'},
                    resumable=MediaFileUpload(str(path), chunksize=1024**2, resumable=True))
                result = None
                while result is None:
                    _, result = request.next_chunk(num_retries=0)
                self.assertEqual(responses, [308, 308, 200])
                self.assertEqual(received, payload)
                self.assertEqual(result['md5Checksum'], hashlib.md5(payload).hexdigest())
        finally:
            server.shutdown()
            thread.join()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
