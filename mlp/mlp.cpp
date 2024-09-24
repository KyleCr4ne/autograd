#include "mlp.hpp"

template <typename T>
SingleNeuron<T>::SingleNeuron(size_t input_size, bool use_activation) {
    use_activation_ = use_activation;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);  

    this->weights_.reserve(input_size);

    for (size_t i = 0; i < input_size; ++i) {
        this->weights_.emplace_back(std::make_shared<Variable<T>>(dis(gen)));
    }
    this->bias_ = std::make_shared<Variable<T>>(dis(gen));
}

template <typename T>
std::shared_ptr<Variable<T>> SingleNeuron<T>::operator()(std::vector<std::shared_ptr<Variable<T>>> &x) {
    std::shared_ptr<Variable<T>> sum = std::make_shared<Variable<T>>(0.0);
    for (size_t i = 0; i < x.size(); ++i) {
        sum = sum + (x[i] * weights_[i]);
    }
    sum = sum + bias_;

    if (use_activation_) return sum->ReLU();
    return sum;
}

template <typename T>
std::vector<std::shared_ptr<Variable<T>>> SingleNeuron<T>::parameters() {
    std::vector<std::shared_ptr<Variable<T>>> params;
    params.reserve(weights_.size() + 1);

    for (auto & weight : weights_) {
        params.emplace_back(weight);
    }
    params.emplace_back(bias_);

    return params;
}

template <typename T>
Linear<T>::Linear(size_t input_size, size_t output_size, bool use_activation) { 
    n_parameters_ = (input_size + 1) * output_size;
    neurons_.reserve(output_size);

    for (size_t i = 0; i < output_size; ++i) {
        SingleNeuron<T> neuron(input_size, use_activation);
        neurons_.emplace_back(neuron);
    }
}

template <typename T>
std::vector<std::shared_ptr<Variable<T>>> Linear<T>::operator()(std::vector<std::shared_ptr<Variable<T>>> x) {
    std::vector<std::shared_ptr<Variable<T>>> output;
    output.reserve(neurons_.size());
    for (auto & neuron : neurons_) {
        output.emplace_back(neuron(x));
    }
    return output;
}

template <typename T>
std::vector<std::shared_ptr<Variable<T>>> Linear<T>::parameters() {
    std::vector<std::shared_ptr<Variable<T>>> params;
    params.reserve(n_parameters_);

    for (auto neuron : neurons_) {
        for (auto weight : neuron.parameters()) {
            params.emplace_back(weight);
        }
    }

    return params;
}

template <typename T>
NN<T>::NN() {
    layers_.reserve(10);
    n_parameters_ = 0;
}

template <typename T>
void NN<T>::add_linear_layer(size_t input_size, size_t output_size, bool use_activation) {
    layers_.emplace_back(Linear(input_size, output_size, use_activation));
    n_parameters_ += (input_size + 1) * output_size;
}

template <typename T>
std::vector<std::shared_ptr<Variable<T>>> NN<T>::operator()(std::vector<std::shared_ptr<Variable<T>>> x) {
    for (auto layer : layers_) {
        x = layer(x);
    }
    return x;
}

template <typename T>
std::vector<std::shared_ptr<Variable<T>>> NN<T>::parameters() {
    std::vector<std::shared_ptr<Variable<T>>> params;
    params.reserve(n_parameters_);

    for (auto layer : layers_) {
        for (auto param : layer.parameters()) {
            params.emplace_back(param);
        }
    }

    return params;
}

template <typename T>
optimizer<T>::optimizer(NN<T> &model, T learning_rate) {
    lr_ = learning_rate;
    model_ = model;
}

template <typename T>
void optimizer<T>::zero_grad() {
    for (auto param : model_.parameters()) {
        param->set_grad(0.0);
    }
}

template <typename T>
void optimizer<T>::step() {
    for (auto param : model_.parameters()) {
        param->set_data(param->get_data_value() - lr_ * param->get_grad_value());
    }
}

