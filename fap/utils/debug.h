// Copyright 2023 The Fap Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef LAB_GEO_APSP_MASTER_FAP_UTILS_DEBUG_H_
#define LAB_GEO_APSP_MASTER_FAP_UTILS_DEBUG_H_

#pragma once

#include <iostream>

using std::cout;
using std::endl;

namespace fap {

template <typename T>
void debug_array(const T *array, const int n) {
    for (int i = 0; i < n; i++) {
        cout << array[i] << " ";
    }
    cout << endl;
}

template <typename T>
void debug_matrix(const T *mat, const int n, const int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            cout << mat[i * m + j] << " ";
        }
        cout << endl;
    }
}

}  // namespace fap

#endif  // LAB_GEO_APSP_MASTER_FAP_UTILS_DEBUG_H_
