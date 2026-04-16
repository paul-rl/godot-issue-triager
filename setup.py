from setuptools import setup, find_packages

setup(
    name="godot-issue-triager",
    version="0.1.0",
    package_dir={"": "src"},           # <-- tells setuptools packages live in src/
    packages=find_packages(where="src"),
    install_requires=["requests"],     
    python_requires=">=3.8",
)
