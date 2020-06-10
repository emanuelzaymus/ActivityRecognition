import time

from src.datasets.Arbua import Aruba
from src.datasets.Kyoto1 import Kyoto1
from src.datasets.Kyoto2 import Kyoto2
from src.datasets.Kyoto3 import Kyoto3
from src.testing.SvmTester import SvmTester

start_time = time.time()

t = SvmTester(Kyoto1())
t.test_default_svm(5)

t = SvmTester(Kyoto2())
t.test_default_svm(5)

t = SvmTester(Kyoto3())
t.test_default_svm(5)

t = SvmTester(Aruba())
t.test_default_svm(5)

print('Execution time: %s s' % (time.time() - start_time))
