from distutils.core import setup

setup(
    name="Ruzicka",
    version="1.1.0",
    author="Ben Nagy",
    packages=["ruzicka"],
    license="MIT",
    url="https://github.com/bnagy/ruzicka",
    long_description_content_type="text/markdown",
    long_description=open("README.md").read(),
    install_requires=[
        "setuptools",
        "scipy",
        "numpy",
        "scikit-learn",
        "numba",
        "typing_extensions",
    ],
)
