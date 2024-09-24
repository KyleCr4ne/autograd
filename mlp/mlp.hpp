#pragma once
#include "../autograd/autograd_variable.cpp"
#include "../autograd/autograd_variable.hpp"
#include <random>


template <typename T = double>
class SingleNeuron {
private:
    std::vector<std::shared_ptr<Variable<T>>> weights_;
    std::shared_ptr<Variable<T>> bias_;
    bool use_activation_;

public:
    SingleNeuron(size_t input_size, bool use_activation=true);
    std::vector<std::shared_ptr<Variable<T>>> parameters();
    std::shared_ptr<Variable<T>> operator()(std::vector<std::shared_ptr<Variable<T>>> &x);
};


template <typename T = double>
class Linear {
private:
    std::vector<SingleNeuron<T>> neurons_;
    size_t n_parameters_;
public:
    Linear(size_t input_size, size_t output_size, bool use_activation=true);
    std::vector<std::shared_ptr<Variable<T>>> operator()(std::vector<std::shared_ptr<Variable<T>>> x);
    std::vector<std::shared_ptr<Variable<T>>> parameters();
};


template <typename T = double>
class NN {
private:
    std::vector<Linear<T>> layers_;
    size_t n_parameters_;
public:
    NN();
    void add_linear_layer(size_t input_size, size_t output_size, bool use_activation);
    std::vector<std::shared_ptr<Variable<T>>> operator()(std::vector<std::shared_ptr<Variable<T>>> x);
    std::vector<std::shared_ptr<Variable<T>>> parameters();
};

template <typename T = double>
class optimizer {
private:
    NN<T> model_;
    T lr_;
public:
    optimizer(NN<T> &model, T learning_rate);
    void zero_grad();
    void step();
};