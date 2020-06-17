import time

from src.classifiers.Preprocessing import Preprocessing
from src.datasets.Aruba import Aruba
from src.datasets.Kyoto1 import Kyoto1
from src.datasets.Kyoto2 import Kyoto2
from src.datasets.Kyoto3 import Kyoto3
from src.testing import TestParameters
from src.testing.SvmTester import SvmTester

start_time = time.time()

# t = SvmTester(Aruba())
# t.test_default_svm()
# t.test_pca()

# for i in [0.9999999999, .99, .95, .9, .85, .8, .75, .7, .6, .5]:
#     # for i in range(5, 40, 5):
#     start_time = time.time()
#     print(i)
#     TestParameters.PCA_N_COMPONENTS = i
#     t.test_default_svm(25)
#     print('Execution time: %s s' % (time.time() - start_time))

t = SvmTester(Aruba(), "last_report/aruba/")

# t.test_pca()

# TestParameters.PREPROCESSOR = Preprocessing.STANDARD_SCALER
# t.test_variable_window_sizes(file_name="window_size_testing-NO_PCA-STANDARD_SCALER", with_time_duration=True)
# TestParameters.PREPROCESSOR = Preprocessing.ROBUST_SCALER
# t.test_variable_window_sizes(file_name="window_size_testing-NO_PCA-ROBUST_SCALER", with_time_duration=True)
# t.test_rbf_c_gamma_parameters(with_time_duration=True)
# t.test_poly_c_gamma_parameters(with_time_duration=True)

t.test_best_svm()

print('Execution time: %s s' % (time.time() - start_time))
