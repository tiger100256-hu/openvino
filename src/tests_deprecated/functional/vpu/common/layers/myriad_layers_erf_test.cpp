// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_erf_test.hpp"

INSTANTIATE_TEST_SUITE_P(
        accuracy, myriadLayersTestsErf_smoke,
        ::testing::ValuesIn(s_ErfDims));
