import time

from src.datasets.aruba.Aruba5Days import Aruba5Days
from src.datasets.kyoto.Kyoto1 import Kyoto1
from src.datasets.kyoto.Kyoto2 import Kyoto2
from src.datasets.kyoto.Kyoto3 import Kyoto3
from src.testing.SvmTester import SvmTester

start_time = time.time()

t = SvmTester(Kyoto1())
t.test_best_svm()

print('Execution time: %s s' % (time.time() - start_time))
