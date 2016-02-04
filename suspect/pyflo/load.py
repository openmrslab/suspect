import pyflo


class TwixComponent(pyflo.component.Component):
    def __init__(self, metadata):
        inports = {
            "filename": {},
        }
        outports = {
            "out": {}
        }
        super().__init__(inports, outports)