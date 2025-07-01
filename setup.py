# pylint: disable=missing-module-docstring
# ruff: noqa

from pathlib import Path

from setuptools import find_packages, setup


def read_long_description() -> str:
    """Return README contents for PyPI description."""
    readme_path = Path(__file__).with_name("README.md")
    return readme_path.read_text(encoding="utf-8") if readme_path.exists() else "FERE Engine"


setup(
    name="fere-engine",
    version="0.1.0",
    description="Twitter trend intelligence pipeline",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    python_requires=">=3.11",
    packages=find_packages(include=["fere_engine", "fere_engine.*"]),
    include_package_data=True,
    install_requires=[
        "pandas>=2.0",
        "httpx>=0.26",
        # keep others in requirements.txt; this is minimal for CLI wrappers
    ],
    entry_points={
        "console_scripts": [
            "fere-general = fere_engine.cli_entrypoints:general",
            "fere-tech = fere_engine.cli_entrypoints:tech",
            "fere-tech-all = fere_engine.cli_entrypoints:tech_all",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 