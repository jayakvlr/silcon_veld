import setuptools


__version__ = "0.0.0"

REPO_NAME = "Silcon :-P"
AUTHOR_USER_NAME = "jayakvlr"
SRC_REPO = "silcon_veld"
AUTHOR_EMAIL = "jayakvlr@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A model MLOPS project for document classification",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{SRC_REPO}.git",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{SRC_REPO}/issues"
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)
