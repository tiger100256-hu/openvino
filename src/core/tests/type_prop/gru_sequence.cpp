// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

struct gru_sequence_parameters {
    Dimension batch_size = 8;
    Dimension num_directions = 1;
    Dimension seq_length = 6;
    Dimension input_size = 4;
    Dimension hidden_size = 128;
    ngraph::element::Type et = element::f32;
};

shared_ptr<opset5::GRUSequence> gru_seq_tensor_initialization(const gru_sequence_parameters& param) {
    auto batch_size = param.batch_size;
    auto seq_length = param.seq_length;
    auto input_size = param.input_size;
    auto num_directions = param.num_directions;
    auto hidden_size = param.hidden_size;
    auto et = param.et;

    const auto X = make_shared<opset5::Parameter>(et, PartialShape{batch_size, seq_length, input_size});
    const auto initial_hidden_state =
        make_shared<opset5::Parameter>(et, PartialShape{batch_size, num_directions, hidden_size});
    const auto sequence_lengths = make_shared<opset5::Parameter>(et, PartialShape{batch_size});
    const auto W = make_shared<opset5::Parameter>(et, PartialShape{num_directions, hidden_size * 3, input_size});
    const auto R = make_shared<opset5::Parameter>(et, PartialShape{num_directions, hidden_size * 3, hidden_size});
    const auto B = make_shared<opset5::Parameter>(et, PartialShape{num_directions, hidden_size * 3});

    const auto gru_sequence = make_shared<opset5::GRUSequence>();

    gru_sequence->set_argument(0, X);
    gru_sequence->set_argument(1, initial_hidden_state);
    gru_sequence->set_argument(2, sequence_lengths);
    gru_sequence->set_argument(3, W);
    gru_sequence->set_argument(4, R);
    gru_sequence->set_argument(5, B);

    return gru_sequence;
}

TEST(type_prop, gru_sequence_forward) {
    const size_t batch_size = 8;
    const size_t num_directions = 1;
    const size_t seq_length = 6;
    const size_t input_size = 4;
    const size_t hidden_size = 128;

    const auto X = make_shared<opset5::Parameter>(element::f32, Shape{batch_size, seq_length, input_size});
    const auto initial_hidden_state =
        make_shared<opset5::Parameter>(element::f32, Shape{batch_size, num_directions, hidden_size});
    const auto sequence_lengths = make_shared<op::Parameter>(element::i32, Shape{batch_size});
    const auto W = make_shared<opset5::Parameter>(element::f32, Shape{num_directions, 3 * hidden_size, input_size});
    const auto R = make_shared<opset5::Parameter>(element::f32, Shape{num_directions, 3 * hidden_size, hidden_size});
    const auto B = make_shared<opset5::Parameter>(element::f32, Shape{num_directions, 3 * hidden_size});

    const auto direction = op::RecurrentSequenceDirection::FORWARD;

    const auto sequence =
        make_shared<opset5::GRUSequence>(X, initial_hidden_state, sequence_lengths, W, R, B, hidden_size, direction);

    EXPECT_EQ(sequence->get_hidden_size(), hidden_size);
    EXPECT_EQ(sequence->get_direction(), op::RecurrentSequenceDirection::FORWARD);
    EXPECT_TRUE(sequence->get_activations_alpha().empty());
    EXPECT_TRUE(sequence->get_activations_beta().empty());
    EXPECT_EQ(sequence->get_activations()[0], "sigmoid");
    EXPECT_EQ(sequence->get_activations()[1], "tanh");
    EXPECT_EQ(sequence->get_clip(), 0.f);
    EXPECT_EQ(sequence->get_linear_before_reset(), false);
    EXPECT_EQ(sequence->get_output_element_type(0), element::f32);
    EXPECT_EQ(sequence->outputs().size(), 2);
    EXPECT_EQ(sequence->get_output_shape(0), (Shape{batch_size, num_directions, seq_length, hidden_size}));
    EXPECT_EQ(sequence->get_output_element_type(1), element::f32);
    EXPECT_EQ(sequence->get_output_shape(1), (Shape{batch_size, num_directions, hidden_size}));
}

TEST(type_prop, gru_sequence_bidirectional) {
    const size_t batch_size = 8;
    const size_t num_directions = 1;
    const size_t seq_length = 6;
    const size_t input_size = 4;
    const size_t hidden_size = 128;

    const auto X = make_shared<opset5::Parameter>(element::f32, Shape{batch_size, seq_length, input_size});
    const auto initial_hidden_state =
        make_shared<opset5::Parameter>(element::f32, Shape{batch_size, num_directions, hidden_size});
    const auto sequence_lengths = make_shared<op::Parameter>(element::i32, Shape{batch_size});
    const auto W = make_shared<opset5::Parameter>(element::f32, Shape{num_directions, 3 * hidden_size, input_size});
    const auto R = make_shared<opset5::Parameter>(element::f32, Shape{num_directions, 3 * hidden_size, hidden_size});
    const auto B = make_shared<opset5::Parameter>(element::f32, Shape{num_directions, 3 * hidden_size});

    const auto direction = op::RecurrentSequenceDirection::BIDIRECTIONAL;
    const std::vector<float> activations_alpha = {2.7, 7.0, 32.367};
    const std::vector<float> activations_beta = {0.0, 5.49, 6.0};
    const std::vector<std::string> activations = {"tanh", "sigmoid"};

    const auto sequence = make_shared<opset5::GRUSequence>(X,
                                                           initial_hidden_state,
                                                           sequence_lengths,
                                                           W,
                                                           R,
                                                           B,
                                                           hidden_size,
                                                           direction,
                                                           activations,
                                                           activations_alpha,
                                                           activations_beta);

    EXPECT_EQ(sequence->get_hidden_size(), hidden_size);
    EXPECT_EQ(sequence->get_direction(), op::RecurrentSequenceDirection::BIDIRECTIONAL);
    EXPECT_EQ(sequence->get_activations_alpha(), activations_alpha);
    EXPECT_EQ(sequence->get_activations_beta(), activations_beta);
    EXPECT_EQ(sequence->get_activations()[0], "tanh");
    EXPECT_EQ(sequence->get_activations()[1], "sigmoid");
    EXPECT_EQ(sequence->get_clip(), 0.f);
    EXPECT_EQ(sequence->get_linear_before_reset(), false);
    EXPECT_EQ(sequence->get_output_element_type(0), element::f32);
    EXPECT_EQ(sequence->outputs().size(), 2);
    EXPECT_EQ(sequence->get_output_shape(0), (Shape{batch_size, num_directions, seq_length, hidden_size}));
    EXPECT_EQ(sequence->get_output_element_type(1), element::f32);
    EXPECT_EQ(sequence->get_output_shape(1), (Shape{batch_size, num_directions, hidden_size}));
}

TEST(type_prop, gru_sequence_dynamic_batch_size) {
    gru_sequence_parameters param;
    param.batch_size = Dimension::dynamic();
    param.num_directions = 2;
    param.seq_length = 6;
    param.input_size = 4;
    param.hidden_size = 128;
    param.et = element::f32;

    auto gru_sequence = gru_seq_tensor_initialization(param);
    gru_sequence->validate_and_infer_types();

    EXPECT_EQ(gru_sequence->get_output_partial_shape(0),
              (PartialShape{param.batch_size, param.num_directions, param.seq_length, param.hidden_size}));
    EXPECT_EQ(gru_sequence->get_output_partial_shape(1),
              (PartialShape{param.batch_size, param.num_directions, param.hidden_size}));
    EXPECT_EQ(gru_sequence->get_output_element_type(0), param.et);
    EXPECT_EQ(gru_sequence->get_output_element_type(1), param.et);
}

TEST(type_prop, gru_sequence_dynamic_num_directions) {
    gru_sequence_parameters param;
    param.batch_size = 8;
    param.num_directions = Dimension::dynamic();
    param.seq_length = 6;
    param.input_size = 4;
    param.hidden_size = 128;
    param.et = element::f32;

    auto gru_sequence = gru_seq_tensor_initialization(param);
    gru_sequence->validate_and_infer_types();

    EXPECT_EQ(gru_sequence->get_output_partial_shape(0),
              (PartialShape{param.batch_size, param.num_directions, param.seq_length, param.hidden_size}));
    EXPECT_EQ(gru_sequence->get_output_partial_shape(1),
              (PartialShape{param.batch_size, param.num_directions, param.hidden_size}));
    EXPECT_EQ(gru_sequence->get_output_element_type(0), param.et);
    EXPECT_EQ(gru_sequence->get_output_element_type(1), param.et);
}

TEST(type_prop, gru_sequence_dynamic_seq_length) {
    gru_sequence_parameters param;
    param.batch_size = 8;
    param.num_directions = 1;
    param.seq_length = Dimension::dynamic();
    param.input_size = 4;
    param.hidden_size = 128;
    param.et = element::f32;

    auto gru_sequence = gru_seq_tensor_initialization(param);
    gru_sequence->validate_and_infer_types();

    EXPECT_EQ(gru_sequence->get_output_partial_shape(0),
              (PartialShape{param.batch_size, param.num_directions, param.seq_length, param.hidden_size}));
    EXPECT_EQ(gru_sequence->get_output_partial_shape(1),
              (PartialShape{param.batch_size, param.num_directions, param.hidden_size}));
    EXPECT_EQ(gru_sequence->get_output_element_type(0), param.et);
    EXPECT_EQ(gru_sequence->get_output_element_type(1), param.et);
}

TEST(type_prop, gru_sequence_dynamic_hidden_size) {
    gru_sequence_parameters param;
    param.batch_size = 8;
    param.num_directions = 1;
    param.seq_length = 6;
    param.input_size = 4;
    param.hidden_size = Dimension::dynamic();
    param.et = element::f32;

    auto gru_sequence = gru_seq_tensor_initialization(param);
    gru_sequence->validate_and_infer_types();

    EXPECT_EQ(gru_sequence->get_output_partial_shape(0),
              (PartialShape{param.batch_size, param.num_directions, param.seq_length, param.hidden_size}));
    EXPECT_EQ(gru_sequence->get_output_partial_shape(1),
              (PartialShape{param.batch_size, param.num_directions, param.hidden_size}));
    EXPECT_EQ(gru_sequence->get_output_element_type(0), param.et);
    EXPECT_EQ(gru_sequence->get_output_element_type(1), param.et);
}

TEST(type_prop, gru_sequence_invalid_input_dimension) {
    gru_sequence_parameters param;

    param.batch_size = 8;
    param.num_directions = 1;
    param.seq_length = 6;
    param.input_size = 4;
    param.hidden_size = 128;
    param.et = element::f32;

    auto gru_sequence = gru_seq_tensor_initialization(param);
    auto invalid_rank0_tensor = make_shared<opset5::Parameter>(param.et, PartialShape{});

    // Validate invalid rank0 tensor for all inputs: X, initial_hidden_state, W, R, B
    for (size_t i = 0; i < gru_sequence->get_input_size(); i++) {
        gru_sequence = gru_seq_tensor_initialization(param);
        gru_sequence->set_argument(i, invalid_rank0_tensor);
        ASSERT_THROW(gru_sequence->validate_and_infer_types(), ngraph::CheckFailure)
            << "GRUSequence node was created with invalid data.";
    }
}

TEST(type_prop, gru_sequence_invalid_input_dynamic_rank) {
    gru_sequence_parameters param;

    param.batch_size = 8;
    param.num_directions = 2;
    param.seq_length = 6;
    param.input_size = 4;
    param.hidden_size = 128;
    param.et = element::f32;

    auto check_dynamic_gru = [](const shared_ptr<opset5::GRUSequence>& gru) -> bool {
        return gru->output(0).get_partial_shape() == PartialShape::dynamic() &&
               gru->output(1).get_partial_shape() == PartialShape::dynamic() &&
               gru->output(0).get_element_type() == gru->input(0).get_element_type();
    };

    auto gru_sequence = gru_seq_tensor_initialization(param);
    auto invalid_dynamic_tensor = make_shared<opset5::Parameter>(param.et, PartialShape::dynamic(Rank::dynamic()));

    // Validate invalid dynamic tensor for all inputs: X, initial_hidden_state, W, R, B
    for (size_t i = 0; i < gru_sequence->get_input_size(); i++) {
        gru_sequence = gru_seq_tensor_initialization(param);
        gru_sequence->set_argument(i, invalid_dynamic_tensor);
        gru_sequence->validate_and_infer_types();
        EXPECT_EQ(check_dynamic_gru(gru_sequence), true);
    }
}
