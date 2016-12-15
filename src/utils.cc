/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "utils.h"

#include <cmath>
#include <ios>

namespace fasttext {

namespace utils {

  int64_t size(std::ifstream& ifs) {
    ifs.seekg(std::streamoff(0), std::ios::end);
    return ifs.tellg();
  }

  void seek(std::ifstream& ifs, int64_t pos) {
    ifs.clear();
    ifs.seekg(std::streampos(pos));
  }

  float softmaxNormalize(float value, float stdevp, float mean) {
    if (stdevp == 0)
      return 0;
    return 1.0 / (1.0 + exp(-1.0 * (value - mean) / stdevp));
  }

  float calculateMean(float data[], int size) {
    if (size == 0)
      return 0;

    float sum = 0.0;
    unsigned int i;

    for (unsigned int i = 0; i < size; i++) {
      sum += data[i];
    }

    return sum / size;
  }

  float calculateStandardDeviation(float data[], int size) {
    if (size == 0)
      return 0;

    float mean, standard_deviation = 0.0;

    mean = calculateMean(data, size);

    for (unsigned int i = 0; i < size; i++) {
      standard_deviation += pow(data[i] - mean, 2);
    }

    return sqrt(standard_deviation / size);
  }

  float calculateStandardDeviation(float data[], int size, float mean) {
    if (size == 0)
      return 0;

    float standard_deviation = 0.0;

    for (unsigned int i = 0; i < size; i++) {
      standard_deviation += pow(data[i] - mean, 2);
    }

    return sqrt(standard_deviation / size);
  }
}

}
