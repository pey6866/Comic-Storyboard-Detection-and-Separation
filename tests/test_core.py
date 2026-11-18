import unittest
from pathlib import Path
import tempfile
from PIL import Image, ImageDraw
from panel_assistant import letterbox, restore, iou, Baseline, process
from evaluate import match


class CoreTests(unittest.TestCase):
    def test_coordinate_roundtrip(self):
        image = Image.new('RGB', (333, 777))
        _, transform = letterbox(image, 640)
        sx, sy, left, top = transform
        box = restore([100*sx+left, 200*sy+top, 100*sx, 200*sy], transform, image.size)
        for actual, expected in zip(box, [50, 100, 100, 200]):
            self.assertLessEqual(abs(actual-expected), 1)

    def test_duplicates_match_once(self):
        self.assertEqual(match([[0, 0, 10, 10]]*2, [[0, 0, 10, 10]], .5), (1, 1, 0))
        self.assertEqual(iou([0, 0, 2, 2], [5, 5, 2, 2]), 0)

    def test_export_and_corrupt_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = Image.new('RGB', (400, 600), 'white')
            ImageDraw.Draw(image).rectangle((20, 20, 380, 280), outline='black', width=4)
            image.save(root/'中文.png')
            output = root/'out'
            output.mkdir()
            params = {'review_threshold': .5}
            record = process(root/'中文.png', output, Baseline(), 'baseline', params)
            self.assertEqual(record['status'], 'needs_review')
            self.assertEqual(len(record['panels']), 1)
            panel = record['panels'][0]
            with Image.open(output/record['page_id']/panel['crop_file']) as crop:
                self.assertEqual(list(crop.size), panel['bbox'][2:])
            (root/'bad.png').write_bytes(b'broken')
            self.assertEqual(process(root/'bad.png', output, Baseline(), 'baseline', params)['status'], 'failed')

    def test_thin_boxes_are_preserved_for_review(self):
        class ThinDetector:
            metadata = {}
            def detect(self, image):
                return [([1, 1, 1, 98], .9)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new('RGB', (100, 100)).save(root/'thin.png')
            record = process(root/'thin.png', root, ThinDetector(), 'inkwell', {'review_threshold': .5})
            self.assertEqual(len(record['panels']), 1)
            self.assertIn('suspicious_thin_box', record['warnings'])
            self.assertFalse(record['human_verified'])


if __name__ == '__main__':
    unittest.main()
