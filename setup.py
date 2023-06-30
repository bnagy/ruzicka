from distutils.core import setup

setup(
    name="Ružička Imposters",
    version="0.0.1",
    author="Ben Nagy",
    packages=["ruzicka-imposters"],
    license="MIT",
    url="https://github.com/bnagy/ruzicka",
    long_description_content_type="text/markdown",
    long_description=open("README.md").read(),
    install_requires=["setuptools", "scipy", "numpy", "scikit-learn", "numba"],
)
