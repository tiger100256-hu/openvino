# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import Mock

try:
    import openvino_telemetry as tm
except ImportError:
    import openvino.tools.mo.utils.telemetry_stub as tm


class UnitTestWithMockedTelemetry(unittest.TestCase):
    def setUp(self):
        tm.Telemetry.__init__ = Mock(return_value=None)
        tm.Telemetry.send_event = Mock()
