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

    std::shared_ptr<Variable<double>> k = x1 * w1;
    k->add_info("k");
    std::shared_ptr<Variable<double>> m = x2 * w2;
    m->add_info("m");
    std::shared_ptr<Variable<double>> t = k + m;
    t->add_info("t");
    std::shared_ptr<Variable<double>> g = t + b;
    g->add_info("g");
    std::shared_ptr<Variable<double>> L = g->Sigmoid();
    L->add_info("L");

    k->get_info();
    m->get_info();
    t->get_info();
    g->get_info();
    L->get_info();

    std::cout << std::endl;
    std::cout << "And for one line:" << std::endl;
    (((x1*w1 + x2*w2) + b)->Sigmoid())->get_info();
    return 0;
}