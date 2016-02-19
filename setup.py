from distutils.core import setup

setup(
        name='suspect',
        version='0.0.1',
        packages=['suspect', 'tests'],
        url='',
        license='MIT',
        author='ben',
        author_email='bennyrowland@mac.com',
        description='',
        entry_points={
        },
        install_requires=['pyflo', 'numpy', 'pytest', 'mock']
)
