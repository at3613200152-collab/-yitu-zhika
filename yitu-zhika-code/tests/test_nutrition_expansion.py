import base64
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.nutrition_video_expansion import choose_object,download_video,scoped,select_frame_indices


class Response:
    def __init__(self,status,headers,chunks):
        self.status_code,self.headers,self.chunks = status,headers,chunks
    def __enter__(self):
        return self
    def __exit__(self,*args):
        return False
    def iter_content(self,chunk_size):
        yield from self.chunks


class ExpansionTests(unittest.TestCase):
    def test_real_short_video_policy(self):
        self.assertEqual(select_frame_indices(47),[0,23,46])
        self.assertEqual(select_frame_indices(61),[0,30,60])
        self.assertEqual(select_frame_indices(100),[0,30,60])
        with self.assertRaises(ValueError):
            select_frame_indices(2)
    def object(self,name='camera_A.h264'):
        return {'name':'nutrition5k_dataset/imagery/side_angles/dish_123/'+name,
                'size':'6','generation':'1234','md5Hash':base64.b64encode(hashlib.md5(b'abcdef').digest()).decode()}
    def test_camera_choice_fixed(self):
        selected = choose_object('dish_123',{'items':[self.object('camera_B.h264'),self.object()]})
        self.assertTrue(selected['name'].endswith('/camera_A.h264'))
    def test_wrong_dish_path_fails(self):
        with self.assertRaises(ValueError):
            choose_object('dish_999',{'items':[self.object()]})
    def test_range_resume_and_generation(self):
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder)/'camera_A.h264'
            video.write_bytes(b'abc')
            response = Response(206,{'Content-Range':'bytes 3-5/6'},[b'def'])
            with patch('scripts.nutrition_video_expansion.requests.get',return_value=response) as get:
                download_video(self.object(),video,lambda:None,lambda **kw:None)
                self.assertEqual(video.read_bytes(),b'abcdef')
                self.assertTrue(get.call_args.args[0].endswith('?generation=1234'))
                self.assertEqual(get.call_args.kwargs['headers']['Range'],'bytes=3-')
    def test_ignored_range_preserves_partial(self):
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder)/'camera_A.h264'
            video.write_bytes(b'abc')
            with patch('scripts.nutrition_video_expansion.requests.get',return_value=Response(200,{'Content-Length':'6'},[b'abcdef'])):
                with self.assertRaises(ValueError):
                    download_video(self.object(),video,lambda:None,lambda **kw:None)
            self.assertEqual(video.read_bytes(),b'abc')
    def test_wrong_total_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder)/'a.h264'
            video.write_bytes(b'abc')
            with patch('scripts.nutrition_video_expansion.requests.get',return_value=Response(206,{'Content-Range':'bytes 3-6/7'},[b'defg'])):
                with self.assertRaises(ValueError):
                    download_video(self.object(),video,lambda:None,lambda **kw:None)
            self.assertEqual(video.read_bytes(),b'abc')
    def test_scope_escape_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                scoped(Path(folder).parent/'escape',folder)


if __name__=='__main__':
    unittest.main()
