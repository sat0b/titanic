#ifndef TITANIC_MODEL_HPP_
#define TITANIC_MODEL_HPP_

#include <cassert>
#include <cmath>
#include <map>
#include <string>
#include <vector>

using StrDataFrame = std::map<std::string, std::vector<std::string>>;
using Vector = std::vector<double>;
using DataFrame = std::map<std::string, Vector>;
using Matrix = std::vector<Vector>;

double average(const Vector &vec) {
  const int n_samples = vec.size();
  double sum = 0.;
  for (auto v : vec) {
    if (std::isnan(v))
      continue;
    sum += v;
  }
  sum /= static_cast<double>(n_samples);
  return sum;
}

double variance(const Vector &vec) {
  double ave = average(vec);
  double var = 0.;
  const int n_samples = vec.size();
  for (auto v : vec)
    var += std::pow(v - ave, 2.);
  var /= static_cast<double>(n_samples - 1);
  return var;
}

void standerize(Vector &vec) {
  double ave = average(vec);
  double std = std::sqrt(variance(vec));
  for (auto &v : vec)
    v = (v - ave) / std;
}

double dot(const Vector &v1, const Vector &v2) {
  int M = v1.size();
  assert(v2.size() == M);
  double out = 0.;
  for (int i = 0; i < M; ++i)
    out += v1[i] * v2[i];
  return out;
}

Vector dot(const Matrix &mat, const Vector &vec) {
  const int N = mat.size();
  const int M = mat[0].size();
  assert(vec.size() == M);

  Vector out(N, 0.);
  for (int i = 0; i < N; ++i)
    out[i] = dot(mat[i], vec);
  return out;
}

double sigmoid(double x) { return 1 / (1 + std::exp(-x)); }

class LogisticRegression {
public:
  explicit LogisticRegression(int n_iter = 50, double alpha = 0.1) {
    this->n_iter = n_iter;
    this->alpha = alpha;
  }

  void fit(const Matrix &matrix, const Vector &targets) {
    const int n_samples = matrix.size();
    const int n_features = matrix[0].size();
    assert(n_samples == targets.size());

    std::fill(weight.begin(), weight.end(), 0.);
    weight.resize(n_features);

    for (int n = 0; n < n_iter; ++n) {
      Vector hx = dot(matrix, weight);
      assert(hx.size() == n_samples);
      for (int i = 0; i < n_features; ++i) {
        double grad = 0.;
        for (int j = 0; j < n_samples; ++j) {
          grad += (hx[j] - targets[j]) * matrix[j][i];
        }
        grad /= static_cast<double>(n_samples);
        weight[i] -= alpha * grad;
      }
    }
  }

  double predict_proba(const Vector &vec) { return sigmoid(dot(weight, vec)); }

  Vector predict_proba(const Matrix &mat) {
    const int N = mat.size();
    Vector y(N, 0);
    for (int i = 0; i < N; ++i)
      y[i] = predict_proba(mat[i]);
    return y;
  }

  double predict(const Vector &vec) {
    return predict_proba(vec) >= 0.5 ? 1. : 0.;
  }

  Vector predict(const Matrix &mat) {
    const int n_samples = mat.size();
    Vector y(n_samples);
    for (int i = 0; i < n_samples; ++i)
      y[i] = predict(mat[i]);
    return y;
  }

  double score(const Matrix &matrix, const Vector &targets) {
    const int n_samples = matrix.size();
    assert(targets.size() == n_samples);
    Vector pred = predict(matrix);
    int count = 0;
    for (int i = 0; i < n_samples; ++i) {
      if (pred[i] == targets[i]) {
        count += 1;
      }
    }
    return static_cast<double>(count) / static_cast<double>(n_samples);
  }

  double cross_validated_score(const Matrix &matrix, const Vector &targets,
                               int n_folds) {
    const int n_samples = matrix.size();
    const int n_sub = n_samples / n_folds;
    Vector scores;

    for (int i = 0; i < n_folds; ++i) {
      // matrix
      Matrix test_matrix(matrix.begin() + i * n_sub,
                         matrix.begin() + (i + 1) * n_sub);
      Matrix train_matrix(matrix.begin(), matrix.begin() + i * n_sub);
      train_matrix.insert(train_matrix.end(), matrix.begin() + (i + 1) * n_sub,
                          matrix.end());
      // targets
      Vector test_targets(targets.begin() + i * n_sub,
                          targets.begin() + (i + 1) * n_sub);
      Vector train_targets(targets.begin(), targets.begin() + i * n_sub);
      train_targets.insert(train_targets.end(),
                           targets.begin() + (i + 1) * n_sub, targets.end());

      // fitting
      fit(train_matrix, train_targets);
      double accuracy = score(test_matrix, test_targets);
      scores.push_back(accuracy);
    }
    return average(scores);
  }

private:
  Vector weight;
  double alpha;
  int n_iter;
};

#endif // TITANIC_MODEL_HPP_
