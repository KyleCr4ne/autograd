#pragma once

#include <vector>
#include <ostream>
#include <memory>
#include <functional>
#include <set>
#include <cmath>
#include <string>
#include <iostream>

/*
 * The main object of autograd engine.
 * Contains information about the current value of the variable,
 * and the partial derivative of the functional for this variable
 * at a fixed value.
 * 
 * Supports basic arithmetic operations and expressions (e.g +,-,*,/,^).
 * 
 * You can get inforamtion about Variable using get_info() method or 
 * using std::cout.
 * 
 * Implemented some of activations functions.
*/

template <typename T = double>
class Variable : public std::enable_shared_from_this<Variable<T>> {
private:
    T data_;
    T grad_;
    std::function<void()> backward_;
    std::set<std::shared_ptr<Variable<T>>> parent_variables_;
    std::string additional_info_;

public:
    Variable(
        T data = 0.0,
        std::set<std::shared_ptr<Variable<T>>> parents = {}
    );

    std::set<std::shared_ptr<Variable<T>>> get_node_parents();
    T get_data_value();
    T get_grad_value();
    void get_info();

    void set_grad(T grad);
    void add_info(const std::string &info);

    void backward();

    std::shared_ptr<Variable<T>> operator+(const std::shared_ptr<Variable<T>> &other);
    std::shared_ptr<Variable<T>> operator-();
    std::shared_ptr<Variable<T>> operator-(const std::shared_ptr<Variable<T>> &other);
    std::shared_ptr<Variable<T>> pow(const std::shared_ptr<Variable<T>> &other);
    std::shared_ptr<Variable<T>> operator/(const std::shared_ptr<Variable<T>> &other);
    std::shared_ptr<Variable<T>> operator*(const std::shared_ptr<Variable<T>> &other);

    std::shared_ptr<Variable<T>> ReLU();
    std::shared_ptr<Variable<T>> Tanh();
    std::shared_ptr<Variable<T>> Sigmoid();
    std::shared_ptr<Variable<T>> exp();
};

template <typename T = double>
std::ostream &operator<<(std::ostream &os, const std::shared_ptr<Variable<T>> &var);

template <typename T = double>
std::shared_ptr<Variable<T>> operator+(const std::shared_ptr<Variable<T>>& lhs, const std::shared_ptr<Variable<T>>& rhs);

template <typename T = double>
std::shared_ptr<Variable<T>> operator-(const std::shared_ptr<Variable<T>>& lhs, const std::shared_ptr<Variable<T>>& rhs);

template <typename T = double>
std::shared_ptr<Variable<T>> operator*(const std::shared_ptr<Variable<T>>& lhs, const std::shared_ptr<Variable<T>>& rhs);

template <typename T = double>
std::shared_ptr<Variable<T>> operator/(const std::shared_ptr<Variable<T>>& lhs, const std::shared_ptr<Variable<T>>& rhs);