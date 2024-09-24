#include "../mlp/mlp.cpp"

double function(double x1, double x2, double x3, double x4) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.01, 0.01);

    return x1 * x2 - x3 + x4 * x4 + dis(gen);
}

std::pair<std::vector<std::shared_ptr<Variable<double>>>, std::shared_ptr<Variable<double>>> gen_sample() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-5.0, 5.0);
    double x1 = dis(gen), x2 = dis(gen), x3 = dis(gen), x4 = dis(gen);
    std::vector<std::shared_ptr<Variable<double>>> sample = {
        std::make_shared<Variable<double>>(x1),
        std::make_shared<Variable<double>>(x2),
        std::make_shared<Variable<double>>(x3),
        std::make_shared<Variable<double>>(x4)
    };
    return std::pair{sample, std::make_shared<Variable<double>>(function(x1, x2, x3, x4))};
}

int main() {
    /*
     * Let's generate a simple data.
     * x1, x2, x3, x4 - > y, where y = f(x1, x2, x3, x4) = x1 * x2 - x3 + x4 ^ 2 + random_noise.
    */
    std::vector<std::vector<std::shared_ptr<Variable<double>>>> X;
    X.reserve(100);
    std::vector<std::shared_ptr<Variable<double>>> Y;
    Y.reserve(100);

    for (int i = 0; i < 100; ++i) {
        auto sample = gen_sample();
        X.emplace_back(sample.first);
        Y.emplace_back(sample.second);
    }

    NN nn; // define model;

    nn.add_linear_layer(4, 10, true); // add linear layer to NN [input_size=4, output_size=10, use_activation=true];
    nn.add_linear_layer(10, 10, true); // add linear layer to NN [input_size=10, output_size=10, use_activation=true];
    nn.add_linear_layer(10, 1, false); // add linear layer to NN [input_size=10, output_size=1, use_activation=false];


    std::cout << "Total trainable parameters: " << nn.parameters().size() << std::endl;

    optimizer<double> optim(nn, 0.0001); // define optimizer;

    size_t epoches = 100;
    
    // Training loop:
    for (size_t i = 0; i < epoches; ++i) {
        double loss_per_epoch = 0.0;
        for (int j = 0; j < 100; ++j) {
            std::vector<std::shared_ptr<Variable<double>>> output = nn(X[j]);
            std::shared_ptr<Variable<double>> loss = Y[j] - output[0];
            loss = loss->pow(std::make_shared<Variable<double>>(2.0));

            loss_per_epoch += loss->get_data_value() / 100.0;

            optim.zero_grad();
            loss->backward();
            optim.step();
        }
        if (i == 0 || (i + 1) % 10 == 0) std::cout << "Epoch " << i + 1 << " Loss: " << loss_per_epoch << std::endl;
    }
    std::cout << std::endl;

    std::vector<std::shared_ptr<Variable<double>>> test_sample = {
        std::make_shared<Variable<double>>(1.0),
        std::make_shared<Variable<double>>(2.0),
        std::make_shared<Variable<double>>(3.0),
        std::make_shared<Variable<double>>(0.5)
    };
    std::vector<std::shared_ptr<Variable<double>>> output = nn(test_sample);
    double expected_output = function(1.0, 2.0, 3.0, 0.5);

    std::cout << "Neural net output: " << output[0]->get_data_value() << ", Expected output: " << expected_output << std::endl;
    return 0;
}