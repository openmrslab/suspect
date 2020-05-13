from setuptools import setup, find_packages

# get the version information from the relevant file
with open('./suspect/_version.py') as f:
    exec(f.read())


setup(
        name='suspect',
        version=__version__,
        packages=find_packages(),
        url='https://github.com/bennyrowland/suspect.git',
        license='MIT',
        author='bennyrowland',
        author_email='bennyrowland@mac.com',
        description='',
        entry_points={
            "console_scripts": [
                "anonymize_twix = suspect.scripts.anonymize:anonymize_twix"
            ]
        },
        classifiers=[
            # How mature is this project? Common values are
            #   3 - Alpha
            #   4 - Beta
            #   5 - Production/Stable
            'Development Status :: 4 - Beta',

            # Indicate who your project is intended for
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Medical Science Apps.',
            'Topic :: Scientific/Engineering :: Physics',

            # Pick your license as you wish (should match "license" above)
            'License :: OSI Approved :: MIT License',

            # Specify the Python versions you support here. In particular, ensure
            # that you indicate whether you support Python 2, Python 3 or both.
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
        install_requires=['pywavelets', 'scipy', 'numpy', 'lmfit', 'pydicom', 'parsley', 'parse', 'nibabel'],
        test_requires=['pytest', 'mock', 'numpydoc']
)
