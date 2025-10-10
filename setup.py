#!/usr/bin/env python3
"""
Setup script for Multi-Modal Cognitive Impairment Assessment (M3-CIA)
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="multimodal-cia",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@institution.edu",
    description="Multi-Modal Cognitive Impairment Assessment using Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zmh56/M3-CIA-Feature-Engineering",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "m3cia-train=scripts.train:main",
            "m3cia-evaluate=scripts.evaluate:main",
            "m3cia-generate-data=scripts.generate_obfuscated_data:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="cognitive-impairment, multi-modal, deep-learning, eeg, ecg, speech, video",
    project_urls={
        "Bug Reports": "https://github.com/zmh56/M3-CIA-Feature-Engineering/issues",
        "Source": "https://github.com/zmh56/M3-CIA-Feature-Engineering",
        "Documentation": "https://m3cia.readthedocs.io/",
    },
)
