class CaptureError(Exception):
    def __init__(self, msg):
        self.msg = msg
        super().__init__(self.msg)

