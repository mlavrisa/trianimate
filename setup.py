from setuptools import setup, find_packages

VERSION = '0.1.1' 
DESCRIPTION = 'Trianimate'
LONG_DESCRIPTION = 'Triangulate images and then animate them!'

# Setting up
setup(
        name="trianimate", 
        version=VERSION,
        author="Matt Lavrisa",
        author_email="m.t.lavrisa@gmail.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[
            'numpy',
            'scipy',
            'opencv',
            'moderngl',
            'pyqt'
        ],
        keywords=['python', 'triangulation', 'animation'],
        classifiers= [
            "Development Status :: 2 - Pre-Alpha",
            "Intended Audience :: End Users/Desktop",
            "Intended Audience :: Other Audience",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: Unix"
        ]
)