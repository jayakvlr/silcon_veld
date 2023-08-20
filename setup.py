import os
from setuptools import find_packages, setup

ROOT = os.path.abspath(os.path.dirname(__file__))


def read_version():
    data = {}
    path = os.path.join(ROOT, "donut", "_version.py")
    with open(path, "r", encoding="utf-8") as f:
        exec(f.read(), data)
    return data["__version__"]


def read_long_description():
    path = os.path.join(ROOT, "README.md")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text
REPO_NAME = "silcon_veld"
AUTHOR_USER_NAME = "jayakvlr"
SRC_REPO = "silcon_veld"
AUTHOR_EMAIL = "jayakvlr@gmail.com"

setup(
    name=REPO_NAME,
    version=read_version(),
    description="OCR-free Document Understanding Transformer",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    url=f"https://github.com/{AUTHOR_USER_NAME}/{SRC_REPO}.git",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{SRC_REPO}/issues"
    },
    package_dir={"": "src"},
    packages=find_packages(
        exclude=[
            "config",
            "dataset",
            "misc",
            "result",
            "synthdog",
            "app.py",
            "lightning_module.py",
            "README.md",
            "train.py",
            "test.py",
        ]
    ),
    python_requires=">=3.7",
    install_requires=[
        "transformers>=4.11.3",
        "timm",
        "datasets[vision]",
        "pytorch-lightning>=1.6.4",
        "nltk",
        "sentencepiece",
        "zss",
        "sconf>=0.2.3",
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)