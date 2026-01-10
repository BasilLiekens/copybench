python_sources = [
    '__init__.py',
]

py3.install_sources(
    python_sources,
    pure: false,
    subdir: 'copybench',
)

py3.extension_module(
    'cppcopy',
    'src/_cpp_copying.cpp',
    install: true,
    dependencies: np_dep,
    subdir: 'copybench'
)
