// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "common_test_utils/include/common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "internal_properties.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/general_utils.h"

using namespace ov::test;
using namespace CPUTestUtils;
using namespace ov::op;

namespace ov {
namespace test {

class PagedAttnQQBiasTest : virtual public ov::test::SubgraphBaseTest, public CPUTestsBase {
public:
    static std::shared_ptr<ov::op::v0::Parameter> make_param(const PartialShape& pshape,
                                                             element::Type element_type,
                                                             const std::string& name) {
        auto param = std::make_shared<v0::Parameter>(element_type, pshape);
        param->set_friendly_name(name);
        param->get_output_tensor(0).set_names({name});
        return param;
    }

    std::shared_ptr<ov::Model> get_pa_model(ov::element::Type data_type,
                                            ov::Dimension::value_type head_size,
                                            ov::Dimension::value_type head_num) {
        auto q = make_param(PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, data_type, "q");
        auto k = make_param(PartialShape{ov::Dimension::dynamic(), head_num * head_size}, data_type, "k");
        auto v = make_param(PartialShape{ov::Dimension::dynamic(), head_num * head_size}, data_type, "v");
        auto key_cache = make_param(PartialShape{ov::Dimension::dynamic(), 32, ov::Dimension::dynamic()},
                                    ov::element::dynamic,
                                    "key_cache.0");
        auto value_cache = make_param(PartialShape{ov::Dimension::dynamic(), 32, ov::Dimension::dynamic()},
                                      ov::element::dynamic,
                                      "value_cache.0");
        auto past_lens = make_param(PartialShape{ov::Dimension::dynamic()}, ov::element::i32, "past_lens");
        auto subsequence_begins =
            make_param(PartialShape{ov::Dimension::dynamic()}, ov::element::i32, "subsequence_begins");
        auto block_indices = make_param(PartialShape{ov::Dimension::dynamic()}, ov::element::i32, "block_indices");
        auto block_indices_begins =
            make_param(PartialShape{ov::Dimension::dynamic()}, ov::element::i32, "block_indices_begins");
        auto qq_bias = make_param(PartialShape{ov::Dimension::dynamic()}, ov::element::u8, "qq_bias");
        auto qq_bias_begins = make_param(PartialShape{ov::Dimension::dynamic()}, ov::element::i32, "qq_bias_begins");

        const float scale_value = 1.0f / std::sqrt(static_cast<float>(head_size));
        auto scale = std::make_shared<v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{scale_value});
        auto sliding_window = std::make_shared<v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{0});
        auto alibi_slopes = std::make_shared<v0::Constant>(ov::element::f32, Shape{0}, std::vector<float>{});
        auto max_context_len = std::make_shared<v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{1024});
        auto score_aggregation_window = std::make_shared<v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{0});
        auto rotated_block_indices = std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{0});
        auto rotation_deltas = std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{0});
        auto rotation_trig_lut = std::make_shared<v0::Constant>(ov::element::f32, Shape{0}, std::vector<float>{0});
        auto xattention_threshold = std::make_shared<v0::Constant>(ov::element::f32, Shape{0}, std::vector<float>{0});
        auto xattention_block_size = std::make_shared<v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{64});
        auto xattention_stride = std::make_shared<v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{8});
        auto sinks = std::static_pointer_cast<v0::Constant>(ov::test::utils::make_constant(data_type, Shape{0}));
        auto adaptive_rkv_start_size = std::make_shared<v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{0});
        auto adaptive_rkv_evictable_sizes = std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{0});
        auto adaptive_rkv_diversity_block_set_indices =
            std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{0});
        auto adaptive_rkv_diversity_block_set_indices_begins =
            std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{0});
        auto token_type_ids = std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{});

        ParameterVector params = {q,
                                  k,
                                  v,
                                  key_cache,
                                  value_cache,
                                  past_lens,
                                  subsequence_begins,
                                  block_indices,
                                  block_indices_begins,
                                  qq_bias,
                                  qq_bias_begins};
        OutputVector pa_inputs = {q,
                                  k,
                                  v,
                                  key_cache,
                                  value_cache,
                                  past_lens,
                                  subsequence_begins,
                                  block_indices,
                                  block_indices_begins,
                                  scale,
                                  sliding_window,
                                  alibi_slopes,
                                  max_context_len,
                                  score_aggregation_window,
                                  rotated_block_indices,
                                  rotation_deltas,
                                  rotation_trig_lut,
                                  xattention_threshold,
                                  xattention_block_size,
                                  xattention_stride,
                                  sinks,
                                  adaptive_rkv_start_size,
                                  adaptive_rkv_evictable_sizes,
                                  adaptive_rkv_diversity_block_set_indices,
                                  adaptive_rkv_diversity_block_set_indices_begins,
                                  token_type_ids,
                                  qq_bias,
                                  qq_bias_begins};

        OPENVINO_ASSERT(pa_inputs.size() == 28);

        auto paged_attn = std::make_shared<op::PagedAttentionExtension>(pa_inputs);
        paged_attn->get_rt_info()["num_k_heads"] = head_num;
        paged_attn->get_rt_info()["k_head_size"] = head_size;
        paged_attn->get_rt_info()["num_v_heads"] = head_num;
        paged_attn->get_rt_info()["v_head_size"] = head_size;

        return std::make_shared<ov::Model>(OutputVector{paged_attn}, params);
    }

    ov::Tensor run_pa_with_qq_bias(const std::vector<uint8_t>& qq_bias_values, const std::vector<int32_t>& qq_bias_begins_values) {
        constexpr size_t seq_len = 3;
        constexpr size_t head_size = 64;
        constexpr size_t head_num = 8;
        const size_t hidden_dim = head_size * head_num;

        targetDevice = ov::test::utils::DEVICE_CPU;
        configuration[ov::hint::inference_precision.name()] = ov::element::f32;
        function = get_pa_model(ov::element::f32, head_size, head_num);
        compile_model();
        auto infer_request = compiledModel.create_infer_request();

        ov::Tensor key_cache_tensor;
        ov::Tensor value_cache_tensor;
        for (const auto& input : compiledModel.inputs()) {
            for (const auto& name : input.get_names()) {
                if (name.find("key_cache.") == 0 || name.find("value_cache.") == 0) {
                    auto pshape = input.get_partial_shape();
                    pshape[0] = 1024 / 32;
                    ov::Tensor cache_tensor(input.get_element_type(), pshape.get_shape());
                    std::memset(cache_tensor.data(), 0, cache_tensor.get_byte_size());
                    if (name.find("key_cache.") == 0) {
                        key_cache_tensor = cache_tensor;
                    } else {
                        value_cache_tensor = cache_tensor;
                    }
                }
            }
        }

        ov::Tensor q_tensor(ov::element::f32, {seq_len, hidden_dim});
        ov::Tensor k_tensor(ov::element::f32, {seq_len, hidden_dim});
        ov::Tensor v_tensor(ov::element::f32, {seq_len, hidden_dim});
        std::memset(q_tensor.data(), 0, q_tensor.get_byte_size());
        std::memset(k_tensor.data(), 0, k_tensor.get_byte_size());

        auto* v_data = v_tensor.data<float>();
        const std::vector<float> token_values = {1.0f, 3.0f, 9.0f};
        for (size_t token = 0; token < seq_len; ++token) {
            std::fill_n(v_data + token * hidden_dim, hidden_dim, token_values[token]);
        }

        ov::Tensor past_lens(ov::element::i32, {1});
        ov::Tensor subsequence_begins(ov::element::i32, {2});
        ov::Tensor block_indices(ov::element::i32, {1});
        ov::Tensor block_indices_begins(ov::element::i32, {2});
        ov::Tensor qq_bias_tensor(ov::element::u8, {qq_bias_values.size()});
        ov::Tensor qq_bias_begins_tensor(ov::element::i32, {qq_bias_begins_values.size()});

        past_lens.data<int32_t>()[0] = 0;
        subsequence_begins.data<int32_t>()[0] = 0;
        subsequence_begins.data<int32_t>()[1] = static_cast<int32_t>(seq_len);
        block_indices.data<int32_t>()[0] = 0;
        block_indices_begins.data<int32_t>()[0] = 0;
        block_indices_begins.data<int32_t>()[1] = 1;
        std::memcpy(qq_bias_tensor.data<uint8_t>(), qq_bias_values.data(), qq_bias_values.size() * sizeof(uint8_t));
        std::memcpy(qq_bias_begins_tensor.data<int32_t>(),
                    qq_bias_begins_values.data(),
                    qq_bias_begins_values.size() * sizeof(int32_t));

        for (auto& param : function->get_parameters()) {
            const auto& name = param->get_friendly_name();
            if (name == "q") {
                infer_request.set_tensor(param, q_tensor);
            } else if (name == "k") {
                infer_request.set_tensor(param, k_tensor);
            } else if (name == "v") {
                infer_request.set_tensor(param, v_tensor);
            } else if (name == "key_cache.0") {
                infer_request.set_tensor(param, key_cache_tensor);
            } else if (name == "value_cache.0") {
                infer_request.set_tensor(param, value_cache_tensor);
            } else if (name == "past_lens") {
                infer_request.set_tensor(param, past_lens);
            } else if (name == "subsequence_begins") {
                infer_request.set_tensor(param, subsequence_begins);
            } else if (name == "block_indices") {
                infer_request.set_tensor(param, block_indices);
            } else if (name == "block_indices_begins") {
                infer_request.set_tensor(param, block_indices_begins);
            } else if (name == "qq_bias") {
                infer_request.set_tensor(param, qq_bias_tensor);
            } else if (name == "qq_bias_begins") {
                infer_request.set_tensor(param, qq_bias_begins_tensor);
            }
        }

        infer_request.infer();

        auto output = infer_request.get_output_tensor(0);
        ov::Tensor output_copy{output.get_element_type(), output.get_shape()};
        output.copy_to(output_copy);
        return output_copy;
    }

    ov::Tensor make_expected_output(const std::vector<float>& expected_token_values) const {
        constexpr size_t head_size = 64;
        constexpr size_t head_num = 8;
        const size_t hidden_dim = head_size * head_num;
        ov::Tensor expected(ov::element::f32, {expected_token_values.size(), hidden_dim});
        auto* expected_data = expected.data<float>();
        for (size_t token = 0; token < expected_token_values.size(); ++token) {
            std::fill_n(expected_data + token * hidden_dim, hidden_dim, expected_token_values[token]);
        }
        return expected;
    }
};

TEST_F(PagedAttnQQBiasTest, MasksCurrentQueriesWithExplicitReference) {
    const auto actual = run_pa_with_qq_bias({
                                               1, 0, 0,
                                               1, 0, 0,
                                               0, 1, 1,
                                           },
                                           {0, 3});

    const auto expected = make_expected_output({1.0f, 1.0f, 6.0f});
    ov::test::utils::compare(expected, actual, 1e-4f, 1e-4f);
}

TEST_F(PagedAttnQQBiasTest, AllOnesPreserveCausalReference) {
    const auto actual = run_pa_with_qq_bias({
                                               1, 1, 1,
                                               1, 1, 1,
                                               1, 1, 1,
                                           },
                                           {0, 3});

    const auto expected = make_expected_output({1.0f, 2.0f, 13.0f / 3.0f});
    ov::test::utils::compare(expected, actual, 1e-4f, 1e-4f);
}

}  // namespace test
}  // namespace ov
