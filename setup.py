"""Setup configuration for LLM Fine-tuning Framework."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="llm-finetuning-framework",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-ready LLM fine-tuning and evaluation framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llm-finetuning-framework",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
            "pre-commit>=2.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-train=scripts.train:main",
            "llm-evaluate=scripts.evaluate:main",
            "llm-api=scripts.run_api:main",
        ],
    },
)