from setuptools import setup, find_packages

VERSION = '0.0.1' 
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
            'scipy'
        ],
        keywords=['python', 'triangulation', 'animation'],
        classifiers= [
            "Development Status :: 1 - Planning",
            "Intended Audience :: End Users/Desktop",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: Unix"
        ]
)