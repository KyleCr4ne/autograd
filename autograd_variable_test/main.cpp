#include "../autograd/autograd_variable.cpp"

int main() {
    /*
     * Let's define several variables:
    */
    std::shared_ptr<Variable<double>> x1 = std::make_shared<Variable<double>>(2.0);
    std::shared_ptr<Variable<double>> x2 = std::make_shared<Variable<double>>(3.0);
    std::shared_ptr<Variable<double>> x3 = std::make_shared<Variable<double>>(5.0);
    std::shared_ptr<Variable<double>> x4 = std::make_shared<Variable<double>>(1.0);
    
    /*
     * Add name info for each:
    */
    x1->add_info("x1");
    x2->add_info("x2");
    x3->add_info("x3");
    x4->add_info("x4");

    x1->get_info();
    x2->get_info();
    x3->get_info();
    x4->get_info();

    std::cout << "std::cout for variable x1: " << x1 << std::endl;
    std::cout << "get data value of variable x1: " << x1->get_data_value() << std::endl;
    std::cout << "get grad value of variable x1: " << x1->get_grad_value() << std::endl;
    
    std::cout << std::endl;

    std::cout << "result of sum x1 + x2 + x3 + x4: " << x1 + x2 + x3 + x4 << std::endl;
    std::cout << "result of productx1 * x2 * x3 * x4: " << x1 * x2 * x3 * x4 << std::endl;
    std::cout << "result of div x1 / x2: " << x1 / x2 << std::endl;
    std::cout << "result of x1 ^ x2: " << x1->pow(x2) << std::endl;
    std::cout << "result of x1 - x2 + x3 " << x1 - x2 + x3 << std::endl;

    std::cout << std::endl;

    std::cout << "result of sigmoid(x1): " << x1->Sigmoid() << std::endl;
    std::cout << "result of exp(x2): " << x2->exp() << std::endl;

    std::cout << std::endl;
    /*
     * And finaly let's calculate some expression and find grad:
    */
    std::shared_ptr<Variable<double>> act = x1 * x2 - x3;
    act->add_info("x1*x2-x3");
    std::shared_ptr<Variable<double>> result = act->Sigmoid();
    result->add_info("sigmoid(act)");
    result->backward();

    
    std::cout << "After backward:" << std::endl;
    x1->get_info();
    x2->get_info();
    x3->get_info();
    x4->get_info();
    result->get_info();

    std::cout << std::endl;

    std::cout << "Parents of result variable:" << std::endl;
    for (auto & parent : result->get_node_parents()) {
        parent->get_info();
    }
    return 0;
}