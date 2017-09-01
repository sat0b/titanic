#include "model.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

using StrDataFrame = std::map<std::string, std::vector<std::string>>;
using Vector = std::vector<double>;
using DataFrame = std::map<std::string, Vector>;
using Matrix = std::vector<Vector>;

void print(const DataFrame &df) {
  for (auto c : df) {
    std::cout << c.first << std::endl;
    for (auto d : c.second)
      std::cout << d << ", ";
    std::cout << std::endl;
  }
}

void print(const Vector &vec) {
  for (auto v : vec)
    std::cout << v << ", ";
  std::cout << std::endl;
}

void print(const Matrix &mat) {
  for (auto c : mat) {
    for (auto r : c) {
      std::cout << r << ", ";
    }
    std::cout << std::endl;
  }
}

std::vector<std::string> split_text(const std::string &str) {
  std::vector<std::string> vec;
  int ii = 0;
  bool escape = false;
  for (int i = 0; i < str.size(); ++i) {
    if (str[i] == '"')
      escape ^= true;
    if (str[i] == ',' && !escape) {
      std::string substr = str.substr(ii, i - ii);
      vec.push_back(str.substr(ii, i - ii));
      ii = i + 1;
    }
  }
  std::string last_word = str.substr(ii);
  if (last_word[last_word.size() - 1 == '\r'] ||
      last_word[last_word.size() - 1 == '\n']) {
    last_word.erase(last_word.length() - 1);
  }
  vec.push_back(last_word);
  return vec;
}

StrDataFrame read_csv(std::string filename) {
  std::ifstream fs(filename);
  std::string buf;
  std::getline(fs, buf);
  std::map<std::string, std::vector<std::string>> df;
  const std::vector<std::string> header = split_text(buf);
  while (!fs.eof()) {
    std::getline(fs, buf);
    if (buf.size() == 0)
      continue;
    std::vector<std::string> sbuf = split_text(buf);
    for (int i = 0; i < header.size(); ++i) {
      df[header[i]].push_back(sbuf[i]);
    }
  }
  return df;
}

Vector stodvec(std::vector<std::string> strvec) {
  Vector vec;
  vec.reserve(strvec.size());
  for (auto v : strvec) {
    if (v.length() == 0)
      vec.push_back(std::numeric_limits<double>::quiet_NaN());
    else
      vec.push_back(std::stod(v));
  }
  return vec;
}

DataFrame one_hot_encoding(const std::vector<std::string> &vec,
                           std::string name) {
  const std::set<std::string> uniq(vec.begin(), vec.end());
  const std::vector<std::string> uniqvec(uniq.begin(), uniq.end());
  const int n_samples = vec.size();
  const int n_features = uniq.size();

  DataFrame df;
  for (int i = 0; i < n_samples; ++i) {
    for (int j = 0; j < n_features; ++j) {
      const std::string header = name + "_" + std::to_string(j);
      if (vec[i] == uniqvec[j]) {
        df[header].push_back(1.);
      } else {
        df[header].push_back(0.);
      }
    }
  }
  return df;
}

void fill_nan(Vector &vec) {
  double sum = 0.;
  for (auto v : vec) {
    if (std::isnan(v))
      continue;
    sum += v;
  }
  sum /= static_cast<double>(vec.size());
  for (auto &v : vec) {
    if (std::isnan(v))
      v = sum;
  }
}

void fill_frequent_word(std::vector<std::string> &vec) {
  std::map<std::string, int> counter;
  for (auto v : vec)
    counter[v] += 1;

  auto max_elm = std::max_element(counter.begin(), counter.end(),
                                  [](const std::pair<std::string, int> &p1,
                                     const std::pair<std::string, int> &p2) {
                                    return p1.second < p2.second;
                                  });
  std::string max_word = max_elm->first;

  for (auto &v : vec) {
    if (v.length() == 0)
      v = max_word;
  }
}

DataFrame make_dataframe(StrDataFrame &strdf) {
  // Initalize
  DataFrame df;

  // Categorical variables
  auto sex = one_hot_encoding(strdf["Sex"], "sex");
  fill_frequent_word(strdf["Embarked"]); // fill nan
  auto v1 = strdf["Embarked"];
  auto s = std::set<std::string>(v1.begin(), v1.end());
  auto embarked = one_hot_encoding(strdf["Embarked"], "embarked");

  // Merge categorical data
  df.insert(sex.begin(), sex.end());
  df.insert(embarked.begin(), embarked.end());

  // Quantitative variables
  std::vector<std::string> headers{"Age", "SibSp", "Parch", "Fare"};
  for (auto header : headers)
    df[header] = stodvec(strdf[header]);

  // Fill nan
  for (auto header : headers)
    fill_nan(df[header]);

  return df;
}

int get_sample_size(const DataFrame &df) {
  std::vector<int> v;
  for (auto kv : df)
    v.push_back(kv.second.size());
  assert(std::set<int>(v.begin(), v.end()).size() == 1);
  return v[0];
}

Matrix df2mat(const DataFrame &df) {
  const int n_samples = df.begin()->second.size();
  const int n_features = df.size();

  // Matrix : m_samples x n_features
  Matrix mat(n_samples, Vector(n_features));
  std::vector<std::string> header;
  for (auto kv : df)
    header.push_back(kv.first);

  for (int i = 0; i < n_samples; ++i) {
    for (int j = 0; j < n_features; ++j) {
      mat[i][j] = df.at(header[j])[i];
    }
  }
  return mat;
}

// Utility
void generate_submission(const Vector &ids, const Vector &predict,
                         const std::string file_path) {
  std::ofstream fp(file_path);
  const int N = ids.size();
  fp << "PassengerId,Survived" << std::endl;
  for (int i = 0; i < N; ++i) {
    fp << ids[i] << "," << predict[i] << std::endl;
  }
  fp.close();
}

int main(int argc, char *argv[]) {

  const std::string train_file_path = "data/train.csv";
  const std::string test_file_path = "data/test.csv";
  const std::string file_path = "submissions/submit.csv";

  // Load data
  auto str_train_df = read_csv(train_file_path);
  auto train_ids = stodvec(str_train_df["PassengerId"]);
  auto train_targets = stodvec(str_train_df["Survived"]);
  auto train_df = make_dataframe(str_train_df);

  auto str_test_df = read_csv(test_file_path);
  auto test_ids = stodvec(str_test_df["PassengerId"]);
  auto test_df = make_dataframe(str_test_df);

  std::vector<std::string> v1;
  for (auto x : train_df)
    v1.push_back(x.first);

  std::vector<std::string> v2;
  for (auto x : test_df)
    v2.push_back(x.first);

  // Standerize dataframe
  for (auto &kv : train_df)
    standerize(kv.second);

  for (auto &kv : test_df)
    standerize(kv.second);

  // Convert to double matrix (n_features x n_samples)
  auto train_matrix = df2mat(train_df);
  auto test_matrix = df2mat(test_df);

  LogisticRegression model;
  const int n_folds = 5;
  double score =
      model.cross_validated_score(train_matrix, train_targets, n_folds);
  std::cout << score << std::endl;

  Vector test_pred = model.predict(test_matrix);
  generate_submission(test_ids, test_pred, file_path);
}
