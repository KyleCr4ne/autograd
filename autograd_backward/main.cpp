#include "../autograd/autograd_variable.cpp"

int main() {
    std::shared_ptr<Variable<double>> x1 = std::make_shared<Variable<double>>(2.0);
    std::shared_ptr<Variable<double>> x2 = std::make_shared<Variable<double>>(1.25);
    std::shared_ptr<Variable<double>> w1 = std::make_shared<Variable<double>>(0.5);
    std::shared_ptr<Variable<double>> w2 = std::make_shared<Variable<double>>(0.75);
    std::shared_ptr<Variable<double>> b = std::make_shared<Variable<double>>(-0.5);

    x1->add_info("x1");
    x2->add_info("x2");
    w1->add_info("w1");
    w2->add_info("w2");
    b->add_info("b");

    std::shared_ptr<Variable<double>> L = (x1 * w1 + x2 * w2 + b)->Sigmoid();
    L->add_info("L");
    L->backward();
    
    x1->get_info();
    x2->get_info();
    w1->get_info();
    w2->get_info();
    L->get_info();
    
    return 0;
}