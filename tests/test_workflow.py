import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import cv2

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workflow import (save_upload, inspect_clip, recent_analyses,
                      process_run, MAX_DURATION_SECONDS, REQUIRED_RESULTS)
from media import ensure_player_video, video_info

DEMO = ROOT / 'outputs' / 'demo'


class NamedBytes(io.BytesIO):
    name = '../../unsafe/name.mp4'


class WorkflowTests(unittest.TestCase):
    def test_demo_video_can_be_inspected(self):
        p = DEMO / 'annotated_boxing_h264.mp4'
        info = inspect_clip(p)
        self.assertAlmostEqual(info.duration, 13.04, delta=0.1)
        self.assertEqual(info.frames, 326)
        self.assertEqual(info.fps, 25)

    def test_save_upload_uses_safe_fixed_path_and_metadata(self):
        with tempfile.TemporaryDirectory() as t:
            up = NamedBytes((DEMO / 'annotated_boxing_h264.mp4').read_bytes())
            run, info = save_upload(up, Path(t))
            self.assertEqual(info.frames, 326)
            self.assertEqual(sorted(p.name for p in run.iterdir()), ['input.mp4', 'run.json'])
            self.assertEqual(json.loads((run / 'run.json').read_text())['original_filename'], 'name.mp4')
            self.assertTrue((run / 'input.mp4').is_file())
            self.assertEqual(recent_analyses(Path(t)), [])  # partial runs must not appear

    def test_bad_video_removes_partial_folder(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):
                save_upload(NamedBytes(b'not a real MP4 video'), Path(t))
            self.assertFalse(list(Path(t).iterdir()))

    def test_successful_processor_results_can_be_reopened(self):
        with tempfile.TemporaryDirectory() as t:
            temp = Path(t)
            run, info = save_upload(NamedBytes((DEMO / 'annotated_boxing_h264.mp4').read_bytes()), temp / 'runs')

            def fake_call(args, **kwargs):
                self.assertFalse(kwargs['shell'])
                self.assertIn('--output-dir', args)
                out = Path(args[args.index('--output-dir') + 1])
                out.mkdir(parents=True, exist_ok=True)
                for name in REQUIRED_RESULTS:
                    source = 'annotated_boxing_h264.mp4' if name == 'annotated_boxing.mp4' else name
                    shutil.copy2(DEMO / source, out / name)
                return type('Completed', (), {'returncode': 0, 'stdout': 'done', 'stderr': ''})()

            with patch('workflow.subprocess.run', side_effect=fake_call):
                result = process_run(run, ROOT, inference_size=960, min_travel=0.34)
            self.assertTrue(result.ok, result.message)
            self.assertEqual(result.output_dir, run / 'outputs')
            options = recent_analyses(temp / 'runs')
            self.assertEqual(len(options), 1)
            self.assertEqual(options[0][1], result.output_dir)
            self.assertEqual(json.loads((run / 'run.json').read_text())['status'], 'complete')

    def test_processor_failure_does_not_appear_as_completed(self):
        with tempfile.TemporaryDirectory() as t:
            temp = Path(t)
            run, _ = save_upload(NamedBytes((DEMO / 'annotated_boxing_h264.mp4').read_bytes()), temp / 'runs')
            with patch('workflow.subprocess.run') as fake:
                fake.return_value = type('Completed', (), {'returncode': 0, 'stdout': '', 'stderr': ''})()
                result = process_run(run, ROOT)
            self.assertFalse(result.ok)
            self.assertEqual(recent_analyses(temp / 'runs'), [])

    def test_browser_player_padding_keeps_hud_above_native_controls(self):
        with tempfile.TemporaryDirectory() as t:
            # Repeat conversion in an isolated directory so tests cover the real FFmpeg filter.
            p = Path(t) / 'fight.mp4'
            shutil.copy2(DEMO / 'annotated_boxing_h264.mp4', p)
            out, warning = ensure_player_video(p)
            if warning and 'not found' in warning.lower():
                self.skipTest('FFmpeg unavailable in test environment')
            self.assertIsNone(warning)
            self.assertTrue(out.exists())
            cap = cv2.VideoCapture(str(out))
            try:
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            finally:
                cap.release()
            self.assertEqual((w, h), (1280, 808))
            self.assertEqual(ensure_player_video(p)[0], out)  # cached


if __name__ == '__main__':
    unittest.main()
