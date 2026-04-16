from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="groupappeals",
    version="1.0.1",  # Removed all obsolete feature references, cleaned documentation
    author="Alona O. Dolinsky; Will Horne; Lena Maria Huber",
    author_email="adolins2@jhu.edu; rwhorne@clemson.edu; lena.huber@uni-mannheim.de",
    description="A toolkit for analyzing group appeals in text using fine-tuned language models optimized for direct plain text processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alonadoli/GroupAppeals",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "groupappeals=groupappeals:cli_main",
        ],
    },
)
