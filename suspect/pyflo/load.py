import pyflo

import suspect.mrs.twix


class TwixComponent(pyflo.component.Component):
    def __init__(self, metadata):
        inports = {
            "filename": {},
        }
        outports = {
            "out": {}
        }
        super().__init__(inports, outports)
        self.inports["filename"].on('data', self.data)

    def data(self, filename):
        data = suspect.mrs.twix.load_twix(filename)
        print("loaded data with shape {} from file {}".format(data.shape, filename))
        self.outports["out"].send_data(data)