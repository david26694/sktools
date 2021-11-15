#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['scikit-learn', 'pandas', 'scipy', 'numpy', 'category_encoders',
                'statsmodels']

setup_requirements = []

misc_requirements = [
    "pip==21.1",
    "bump2version==0.5.11",
    "wheel==0.33.6",
    "watchdog==0.9.0",
    "flake8==3.7.8",
    "tox==3.14.0",
    "coverage==4.5.4",
    "Sphinx==1.8.5",
    "sphinx-rtd-theme==0.4.3",
    "twine==1.14.0",
    "pre-commit==2.6.0",
]

test_requirements = requirements

dev_requirements = misc_requirements + requirements

setup(
    author="David Masip Bonet",
    author_email='david26694@gmail.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Tools to extend sklearn",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='sktools',
    name='sktools',
    packages=find_packages(include=['sktools', 'sktools.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    extras_require={
        "test": test_requirements,
        "dev": dev_requirements
    },
    url='https://github.com/david26694/sktools',
    version='0.1.4',
    zip_safe=False,
)
