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
            "pyflo": [
                "pyflo.mrs.load.twix = suspect.pyflo.load:TwixComponent",
                "pyflo.mrs.svdchannel = suspect.pyflo.processing:SVDChannelWeights",
                "pyflo.mrs.weightedaverage = suspect.pyflo.processing:WeightedAverage",
                "pyflo.mrs.frequency_correction.water_peak_alignment = suspect.pyflo.processing:WaterPeakAlignment",
                "pyflo.mrs.frequency_correction.frequency_shift = suspect.pyflo.processing:FrequencyShift",
                "pyflo.mrs.plot = suspect.pyflo.processing:TempPlot",
            ]
        },
        install_requires=['pyflo', 'numpy', 'pytest', 'mock']
)
