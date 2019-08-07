import os
import unittest
from shutil import rmtree


# TODO need a knossos example file
class TestKnossosWrapper(unittest.TestCase):
    tmp_dir = './tmp'

    def setUp(self):
        os.makedirs(self.tmp_dir)

    def tearDown(self):
        try:
            rmtree(self.tmp_dir)
        except OSError:
            pass

    def test_import(self):
        from elf.io.knossos_wrapper import KnossosFile
        from elf.io.knossos_wrapper import KnossosDataset


if __name__ == '__main__':
    unittest.main()
