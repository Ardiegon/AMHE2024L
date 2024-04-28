from setuptools import setup, find_packages

setup(
    name="amhepr4",
    version="0.1.0",
    description="des and nl shade comparison and merge",
    author="Oskar Bartosz, Michał Matak",
    packages=find_packages("src"),  # Find packages in src directory
    package_dir={"": "src"},  # Treat "src" directory as the source of packages
    install_requires=[
        'matplotlib',
        'numpy',
        "tmux"
    ],
    entry_points={
        "console_scripts": []
    },
)