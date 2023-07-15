import os
import subprocess
import unittest


# code testing and debugging during development.


class TestYourScript(unittest.TestCase):
    def test_export(self):
        # vit_b
        subprocess.check_call(['python', 'main.py', 'export', '--model_path', 'pth_model/sam_vit_b_01ec64.pth', '--model_precision', 'fp32'])
        self.assertTrue(os.path.isfile('exported_models/vit_b/model_fp32.engine'))
        subprocess.check_call(['python', 'main.py', 'export', '--model_path', 'pth_model/sam_vit_b_01ec64.pth', '--model_precision', 'fp16'])
        self.assertTrue(os.path.isfile('exported_models/vit_b/model_fp16.engine'))
        os.remove('exported_models/vit_b/model_fp32.engine')
        os.remove('exported_models/vit_b/model_fp16.engine')
        subprocess.check_call(['python', 'main.py', 'export', '--model_path', 'pth_model/sam_vit_b_01ec64.pth', '--model_precision', 'both'])
        self.assertTrue(os.path.isfile('exported_models/vit_b/model_fp32.engine'))
        self.assertTrue(os.path.isfile('exported_models/vit_b/model_fp16.engine'))

        # vit_l
        subprocess.check_call(['python', 'main.py', 'export', '--model_path', 'pth_model/sam_vit_l_0b3195.pth', '--model_precision', 'fp32'])
        self.assertTrue(os.path.isfile('exported_models/vit_l/model_fp32.engine'))
        subprocess.check_call(['python', 'main.py', 'export', '--model_path', 'pth_model/sam_vit_l_0b3195.pth', '--model_precision', 'fp16'])
        self.assertTrue(os.path.isfile('exported_models/vit_l/model_fp16.engine'))
        os.remove('exported_models/vit_l/model_fp32.engine')
        os.remove('exported_models/vit_l/model_fp16.engine')
        subprocess.check_call(['python', 'main.py', 'export', '--model_path', 'pth_model/sam_vit_l_0b3195.pth', '--model_precision', 'both'])
        self.assertTrue(os.path.isfile('exported_models/vit_l/model_fp32.engine'))
        self.assertTrue(os.path.isfile('exported_models/vit_l/model_fp16.engine'))

        # vit_h
        subprocess.check_call(['python', 'main.py', 'export', '--model_path', 'pth_model/sam_vit_h_4b8939.pth', '--model_precision', 'fp32'])
        self.assertTrue(os.path.isfile('exported_models/vit_h/model_fp32_1.engine'))
        self.assertTrue(os.path.isfile('exported_models/vit_h/model_fp32_2.engine'))
        subprocess.check_call(['python', 'main.py', 'export', '--model_path', 'pth_model/sam_vit_h_4b8939.pth', '--model_precision', 'fp16'])
        self.assertTrue(os.path.isfile('exported_models/vit_h/model_fp16_1.engine'))
        self.assertTrue(os.path.isfile('exported_models/vit_h/model_fp16_2.engine'))
        os.remove('exported_models/vit_h/model_fp32_1.engine')
        os.remove('exported_models/vit_h/model_fp16_1.engine')
        os.remove('exported_models/vit_h/model_fp32_2.engine')
        os.remove('exported_models/vit_h/model_fp16_2.engine')
        subprocess.check_call(['python', 'main.py', 'export', '--model_path', 'pth_model/sam_vit_h_4b8939.pth', '--model_precision', 'both'])
        self.assertTrue(os.path.isfile('exported_models/vit_h/model_fp32_1.engine'))
        self.assertTrue(os.path.isfile('exported_models/vit_h/model_fp16_1.engine'))
        self.assertTrue(os.path.isfile('exported_models/vit_h/model_fp16_2.engine'))
        self.assertTrue(os.path.isfile('exported_models/vit_h/model_fp32_2.engine'))

    def test_benchmark(self):
        # vit_b
        subprocess.check_call(['python', 'main.py', 'benchmark', '--sam_checkpoint', 'pth_model/sam_vit_b_01ec64.pth', '--model_type', 'vit_b', '--warmup_iters', '5', '--measure_iters', '50'])
        # vit_l
        subprocess.check_call(['python', 'main.py', 'benchmark', '--sam_checkpoint', 'pth_model/sam_vit_l_0b3195.pth', '--model_type', 'vit_l', '--warmup_iters', '5', '--measure_iters', '50'])
        # vit_h
        subprocess.check_call(['python', 'main.py', 'benchmark', '--sam_checkpoint', 'pth_model/sam_vit_h_4b8939.pth', '--model_type', 'vit_h', '--warmup_iters', '5', '--measure_iters', '50'])
        # test all
        subprocess.check_call(['python', 'main.py', 'benchmark', '--sam_checkpoint', 'pth_model', '--model_type', 'all'])

    def test_infer(self):
        # vit_b
        subprocess.check_call(['python', 'main.py', 'infer', '--pth_path', 'pth_model/sam_vit_b_01ec64.pth', '--model_1', 'exported_models/vit_b/model_fp32.engine', '--img_path', 'images/original_image.jpg'])
        subprocess.check_call(['python', 'main.py', 'infer', '--pth_path', 'pth_model/sam_vit_b_01ec64.pth', '--model_1', 'exported_models/vit_b/model_fp16.engine', '--img_path', 'images/original_image.jpg'])
        # vit_l
        subprocess.check_call(['python', 'main.py', 'infer', '--pth_path', 'pth_model/sam_vit_l_0b3195.pth', '--model_1', 'exported_models/vit_l/model_fp32.engine', '--img_path', 'images/original_image.jpg'])
        subprocess.check_call(['python', 'main.py', 'infer', '--pth_path', 'pth_model/sam_vit_l_0b3195.pth', '--model_1', 'exported_models/vit_l/model_fp16.engine', '--img_path', 'images/original_image.jpg'])
        # vit_h
        subprocess.check_call(['python', 'main.py', 'infer', '--pth_path', 'pth_model/sam_vit_h_4b8939.pth', '--model_1', 'exported_models/vit_h/model_fp32_1.engine', '--model_2', 'exported_models/vit_h/model_fp32_2.engine', '--img_path', 'images/original_image.jpg'])
        subprocess.check_call(['python', 'main.py', 'infer', '--pth_path', 'pth_model/sam_vit_h_4b8939.pth', '--model_1', 'exported_models/vit_h/model_fp16_1.engine', '--model_2', 'exported_models/vit_h/model_fp16_2.engine', '--img_path', 'images/original_image.jpg'])


    def test_accuracy(self):
        # vit_b
        subprocess.check_call(['python', 'main.py', 'accuracy', '--image_dir', 'test_images', '--model_type', 'vit_b', '--sam_checkpoint', 'pth_model/sam_vit_b_01ec64.pth'])
        # vit_l
        subprocess.check_call(['python', 'main.py', 'accuracy', '--image_dir', 'test_images', '--model_type', 'vit_l', '--sam_checkpoint', 'pth_model/sam_vit_l_0b3195.pth'])
        # vit_h
        subprocess.check_call(['python', 'main.py', 'accuracy', '--image_dir', 'test_images', '--model_type', 'vit_h', '--sam_checkpoint', 'pth_model/sam_vit_h_4b8939.pth'])
        # test all
        subprocess.check_call(['python', 'main.py', 'accuracy', '--image_dir', 'test_images', '--model_type', 'all', '--sam_checkpoint', 'pth_model'])
        # test all
        subprocess.check_call(['python', 'main.py', 'accuracy', '--image_dir', 'test_images', '--model_type', 'all', '--sam_checkpoint', 'pth_model', '--save'])



if __name__ == '__main__':
    class CustomTestSuite(unittest.TestSuite):
        def __init__(self, tests=()):
            super().__init__(tests)
            self.test_order = [
                TestYourScript.test_export,
                TestYourScript.test_benchmark,
                TestYourScript.test_accuracy,
                TestYourScript.test_infer
            ]

        def addTest(self, test):
            # Ensure the tests are added in the desired order
            for test_func in self.test_order:
                if test_func.__name__ == test._testMethodName:
                    super().addTest(test)
                    break


    suite = CustomTestSuite()
    suite.addTest(TestYourScript('test_export'))
    suite.addTest(TestYourScript('test_benchmark'))
    suite.addTest(TestYourScript('test_accuracy'))
    suite.addTest(TestYourScript('test_infer'))

    unittest.TextTestRunner().run(suite)
