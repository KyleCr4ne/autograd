#include "autograd_variable.hpp"

template <typename T>
Variable<T>::Variable(
    T data,
    std::set<std::shared_ptr<Variable<T>>> parents) {
    
    this->data_ = data;
    this->grad_ = 0.0;
    this->parent_variables_ = std::move(parents);
    this->backward_ = []() {};
    this->additional_info_ = "";
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::shared_ptr<Variable<T>> &var) {
    return os << "Variable(data=" << var->get_data_value() << ", grad=" << var->get_grad_value() << ")";
}

template <typename T>
std::set<std::shared_ptr<Variable<T>>> Variable<T>::get_node_parents() {
    return parent_variables_;
}

template <typename T>
T Variable<T>::get_data_value() {
    return data_;
}

template <typename T>
T Variable<T>::get_grad_value() {
    return grad_;
}

template <typename T>
void Variable<T>::get_info() {
    std::cout << "Variable(data=" << data_ << ", grad=" << grad_ << ", info=" << additional_info_ << ")" << std::endl;
}

template <typename T>
void Variable<T>::set_grad(T grad) {
    this->grad_ = grad;
}

template <typename T>
void Variable<T>::set_data(T data) {
    this->data_ = data;
}

template <typename T>
void Variable<T>::add_info(const std::string &info) {
    additional_info_ = std::move(info);
}

template <typename T>
void Variable<T>::backward() {
    grad_ = 1.0;
    std::vector<std::shared_ptr<Variable<T>>> order;
    std::set<std::shared_ptr<Variable<T>>> visited;

    std::function<void(const std::shared_ptr<Variable<T>>&)> dfs = 
    [&] (const std::shared_ptr<Variable<T>> &variable) {
        if (visited.count(variable) > 0) return;
        visited.insert(variable);
        for (auto & parent : variable->parent_variables_) {
            dfs(parent);
        }
        order.push_back(variable);
    };

    dfs(this->shared_from_this());

    for (int i = (int) order.size() - 1; i >= 0; --i) {
        order[i]->backward_();
    }
}

template <typename T>
std::shared_ptr<Variable<T>> Variable<T>::operator+(const std::shared_ptr<Variable<T>> &other) {
    auto result = std::make_shared<Variable<T>>(
        data_ + other->data_,
        std::set<std::shared_ptr<Variable<T>>>{
            this->shared_from_this(),
            other
        }
    );

    result->backward_ = [this, other, result]() {
        grad_ += result->grad_;
        other->grad_ += result->grad_;
    };

    return result;
}

template <typename T>
std::shared_ptr<Variable<T>> Variable<T>::operator*(const std::shared_ptr<Variable<T>> &other) {
    auto result = std::make_shared<Variable<T>>(
        data_ * other->data_,
        std::set<std::shared_ptr<Variable<T>>>{
            this->shared_from_this(),
            other
        }
    );

    result->backward_ = [this, other, result]() {
        grad_ += other->data_ * result->grad_;
        other->grad_ += data_ * result->grad_;
    };

    return result;
}

template <typename T>
std::shared_ptr<Variable<T>> Variable<T>::pow(const std::shared_ptr<Variable<T>> &other) {
    auto result = std::make_shared<Variable<T>>(
        std::pow(data_, other->data_),
        std::set<std::shared_ptr<Variable<T>>>{
            this->shared_from_this(),
            other
        }
    );

    result->backward_ = [this, other, result]() {
        grad_ += other->data_ * std::pow(data_, other->data_ - 1.0) * result->grad_;
    };

    return result;
}

template <typename T>
std::shared_ptr<Variable<T>> Variable<T>::ReLU() {
    auto result = std::make_shared<Variable<T>>(
        data_ < 0.0 ? 0.0 : data_,
        std::set<std::shared_ptr<Variable<T>>>{
            this->shared_from_this()
        }
    );

    result->backward_ = [this, result]() {
        grad_ += (result->data_ > 0.0) * result->grad_;
    };

    return result;
}

template <typename T>
std::shared_ptr<Variable<T>> Variable<T>::Tanh() {
    auto result = std::make_shared<Variable<T>>(
        (std::exp(2 * data_) - 1.0) / (std::exp(2 * data_) + 1.0),
        std::set<std::shared_ptr<Variable<T>>>{
            this->shared_from_this()
        }
    );

    result->backward_ = [this, result]() {
        grad_ += (1.0 - result->data_ * result->data_) * result->grad_;
    };

    return result;
}

template <typename T>
std::shared_ptr<Variable<T>> Variable<T>::Sigmoid() {
    auto result = std::make_shared<Variable<T>>(
        1.0 / (1.0 + std::exp(-data_)),
        std::set<std::shared_ptr<Variable<T>>>{
            this->shared_from_this()
        }
    );

    result->backward_ = [this, result]() {
        grad_ += (result->data_ * (1.0 - result->data_)) * result->grad_;
    };

    return result;
}

template <typename T>
std::shared_ptr<Variable<T>> Variable<T>::exp() {
    auto result = std::make_shared<Variable<T>>(
        std::exp(data_),
        std::set<std::shared_ptr<Variable<T>>>{
            this->shared_from_this()
        }
    );

    result->backward_ = [this, result]() {
        grad_ += result->data_ * result->grad_;
    };

    return result;
}

template <typename T>
std::shared_ptr<Variable<T>> Variable<T>::operator-() {
    return this->shared_from_this() * std::make_shared<Variable<T>>(-1.0);
}

template <typename T>
std::shared_ptr<Variable<T>> Variable<T>::operator-(const std::shared_ptr<Variable<T>> &other) {
    return this->shared_from_this() + (other->operator-());
}

template <typename T>
std::shared_ptr<Variable<T>> Variable<T>::operator/(const std::shared_ptr<Variable<T>> &other) {
    return this->shared_from_this() * other->pow(std::make_shared<Variable<T>>(-1.0));
}


template <typename T>
std::shared_ptr<Variable<T>> operator+(const std::shared_ptr<Variable<T>>& lhs, const std::shared_ptr<Variable<T>>& rhs) {
    return (*lhs) + rhs;
}

template <typename T>
std::shared_ptr<Variable<T>> operator-(const std::shared_ptr<Variable<T>>& lhs, const std::shared_ptr<Variable<T>>& rhs) {
    return (*lhs) - rhs;
}

template <typename T>
std::shared_ptr<Variable<T>> operator*(const std::shared_ptr<Variable<T>>& lhs, const std::shared_ptr<Variable<T>>& rhs) {
    return (*lhs) * rhs;
}

template <typename T>
std::shared_ptr<Variable<T>> operator/(const std::shared_ptr<Variable<T>>& lhs, const std::shared_ptr<Variable<T>>& rhs) {
    return (*lhs) / rhs;
}