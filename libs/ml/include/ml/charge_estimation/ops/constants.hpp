#pragma once
//------------------------------------------------------------------------------
//
//   Copyright 2018-2020 Fetch.AI Limited
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//
//------------------------------------------------------------------------------

#include "ml/charge_estimation/types.hpp"

namespace fetch {
namespace ml {
namespace ops {
namespace charge_cost {

static constexpr MLChargeAmount FIBONNACI_GENERATOR_PER_ELEMENT = 1;
static constexpr MLChargeAmount TANH_PER_ELEMENT                = 1;
static constexpr MLChargeAmount MAX_PER_ELEMENT                 = 1;
static constexpr MLChargeAmount ADDITION_PER_ELEMENT            = 1;
static constexpr MLChargeAmount SUBTRACTION_PER_ELEMENT         = ADDITION_PER_ELEMENT;
static constexpr MLChargeAmount FLATTEN_PER_ELEMENT             = 1;
static constexpr MLChargeAmount EXP_PER_ELEMENT                 = 1;
static constexpr MLChargeAmount LOG_PER_ELEMENT                 = 1;
static constexpr MLChargeAmount MULTIPLICATION_PER_ELEMENT      = 3;
static constexpr MLChargeAmount DIVISION_PER_ELEMENT            = 1;
static constexpr MLChargeAmount PLACEHOLDER_READING_PER_ELEMENT = 0;
static constexpr MLChargeAmount WEIGHTS_READING_PER_ELEMENT     = 0;
static constexpr MLChargeAmount POW_PER_ELEMENT                 = EXP_PER_ELEMENT + LOG_PER_ELEMENT + MULTIPLICATION_PER_ELEMENT;
static constexpr MLChargeAmount SIGMOID_PER_ELEMENT             = EXP_PER_ELEMENT + ADDITION_PER_ELEMENT + DIVISION_PER_ELEMENT;
static constexpr MLChargeAmount LOG_SIGMOID_PER_ELEMENT         = SIGMOID_PER_ELEMENT + LOG_PER_ELEMENT;
static constexpr MLChargeAmount RELU_PER_ELEMENT                = MAX_PER_ELEMENT;
static constexpr MLChargeAmount DROPOUT_PER_ELEMENT             = FIBONNACI_GENERATOR_PER_ELEMENT + SUBTRACTION_PER_ELEMENT + DIVISION_PER_ELEMENT + MULTIPLICATION_PER_ELEMENT ;
static constexpr MLChargeAmount ELU_PER_ELEMENT                 = EXP_PER_ELEMENT + SUBTRACTION_PER_ELEMENT + MULTIPLICATION_PER_ELEMENT;
static constexpr MLChargeAmount GELU_PER_ELEMENT                = MULTIPLICATION_PER_ELEMENT + POW_PER_ELEMENT + MULTIPLICATION_PER_ELEMENT + \
		 + ADDITION_PER_ELEMENT + TANH_PER_ELEMENT + ADDITION_PER_ELEMENT + MULTIPLICATION_PER_ELEMENT + MULTIPLICATION_PER_ELEMENT;
static constexpr MLChargeAmount LEAKY_RELU_PER_ELEMENT           = MAX_PER_ELEMENT + MULTIPLICATION_PER_ELEMENT;
static constexpr MLChargeAmount RANDOMISED_RELU_PER_ELEMENT      = FIBONNACI_GENERATOR_PER_ELEMENT + MAX_PER_ELEMENT + MULTIPLICATION_PER_ELEMENT;
static constexpr MLChargeAmount SOFTMAX_PER_ELEMENT              = MAX_PER_ELEMENT + SUBTRACTION_PER_ELEMENT + EXP_PER_ELEMENT + ADDITION_PER_ELEMENT + DIVISION_PER_ELEMENT;
static constexpr MLChargeAmount LOG_SOFTMAX_PER_ELEMENT          = LOG_PER_ELEMENT + SOFTMAX_PER_ELEMENT;
}  // namespace charge_cost
}  // namespace ops
}  // namespace ml
}  // namespace fetch
