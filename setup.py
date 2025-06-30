from setuptools import setup, find_packages

# Lisez le contenu de requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="moge_viewer",
    version="1.0",
    author="Votre Nom",
    description="Un visualiseur 3D interactif pour les modèles MoGe de Microsoft.",
    long_description="""
    Cette application utilise le modèle MoGe pour estimer la géométrie 3D
    d'une image et l'affiche dans un visualiseur interactif.
    L'installation est gérée par des scripts personnalisés pour installer
    automatiquement la version correcte de PyTorch avec support CUDA.
    """,
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=required,
    entry_points={
        'console_scripts': [
            'moge_viewer=main:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)