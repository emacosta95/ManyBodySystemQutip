from setuptools import setup, find_packages

setup(
    name="ManyBodyQutip",
    version="0.1.0",
    description="A library that wraps QuTiP to simulate many-body quantum systems",
    author="Emanuele Costa",
    packages=find_packages(),  # looks in current directory
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "qutip"
    ],
    python_requires=">=3.8",
)