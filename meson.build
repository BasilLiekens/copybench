project(
    'copybench',
    'cpp',
    version: '0.0.1',
    license: 'MIT',
    meson_version: '>= 1.3.0',
    default_options: [
    'buildtype=debugoptimized',
    'b_ndebug=if-release',
    'cpp_std=c++17',
  ],
)

py3 = import('python').find_installation(pure: false)
py3_dep = py3.dependency()

cpp = meson.get_compiler('cpp')


# Yanked from scipy
incdir_numpy = meson.get_external_property('numpy-include-dir', 'not-given')
if incdir_numpy == 'not-given'
  incdir_numpy = run_command(py3,
    [
      '-c',
      '''
import os
import numpy as np
try:
  incdir = os.path.relpath(np.get_include())
except Exception:
  incdir = np.get_include()
print(incdir)
  '''
    ],
    check: true
  ).stdout().strip()

  _incdir_numpy_abs = run_command(py3,
    ['-c', 'import numpy; print(numpy.get_include())'],
    check: true
  ).stdout().strip()
else
  _incdir_numpy_abs = incdir_numpy
endif
inc_np = include_directories(incdir_numpy)

numpy_nodepr_api = ['-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION']
np_dep = declare_dependency(include_directories: inc_np, compile_args: numpy_nodepr_api)

subdir('src')
