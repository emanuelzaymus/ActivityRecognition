import time

from src.datasets.Aruba import Aruba
from src.datasets.Kyoto1 import Kyoto1
from src.datasets.Kyoto2 import Kyoto2
from src.datasets.Kyoto3 import Kyoto3
from src.testing import TestParameters
from src.testing.SvmTester import SvmTester

# start_time = time.time()

t = SvmTester(Aruba())
for i in [0.9999999999, .99, .95, .9, .85, .8, .75, .7, .6, .5]:
    # for i in range(5, 40, 5):
    start_time = time.time()
    print(i)
    TestParameters.PCA_N_COMPONENTS = i
    t.test_default_svm(25)
    print('Execution time: %s s' % (time.time() - start_time))

# t = SvmTester(Kyoto2())
# t.test_default_svm(5)
#
# t = SvmTester(Kyoto3())
# t.test_default_svm(5)
#
# t = SvmTester(Aruba())
# t.test_default_svm(5)

# print('Execution time: %s s' % (time.time() - start_time))
