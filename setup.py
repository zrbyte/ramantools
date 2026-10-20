from setuptools import setup, find_packages

setup(
    name = "ramantools",
    version = "0.6.0",
    packages = find_packages(),
    # python_requires='>=3.10',
    install_requires = [
        "matplotlib",
        "numpy",
        "pandas",
        "scipy",
        "xarray>=2023.7.0",
        # Reads WITec project files (.wip / .wid) for rt.wipfile.
        "WITecSDK>=2.0.3",
        # netCDF4/HDF5 backend for ramanmap.save() / singlespec.save() / rt.load().
        "h5netcdf"
    ],
    # ``ramantools`` command-line tool: inspect / export / plot without
    # writing Python (see ramantools/cli.py).
    entry_points = {
        "console_scripts": ["ramantools = ramantools.cli:main"],
    },
    description = "Tools for analyzing Raman spectroscopy data, measured by a Witec confocal Raman microscope.",
    long_description = open('README.md', 'r').read(),
    long_description_content_type = "text/markdown",
    author = "Peter Nemes-Incze",
    url = "https://github.com/zrbyte/ramantools",
    classifiers = [
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
)
