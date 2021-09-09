from setuptools import setup

setup(
    name="lila",
    version="0.1.0",
    py_modules=["lila"],
    install_requires=[
        "scikit-bot==0.6.1",
        "numpy==1.21.2",
    ],
    entry_points={
        "console_scripts": [
            "lila = lila:cli",
        ],
    },
)
