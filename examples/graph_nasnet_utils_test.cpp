
#include "graph_nasnet_utils.h"

bool test_operation_to_info(){
    std::vector<std::string> operations = {
                "separable_5x5_2",
                "separable_3x3_2",
                "separable_7x7_4",
            };
    std::vector<std::vector<int>> expect = {{2, 5}, {2, 3}, {4, 7}};
    for(size_t i=0; i<operations.size(); ++i){
        if(_operation_to_info(operations[i]) != expect[i]){
            return false;
        }
    }
    return true;
}

bool test_operation_to_pooling_info(){
    std::vector<std::string> operations = {
                "avg_pool_3x3",
                "max_pool_5x5"
    };
    std::vector<std::vector<std::string>> expect = {{"avg", "3"}, {"max", "5"}};
    for(size_t i=0; i<operations.size(); ++i){
        if(_operation_to_pooling_info(operations[i]) != expect[i]){
            return false;
        }
    }
    return true;
}

bool test_calc_reduction_layers(){
    auto result = calc_reduction_layers(18, 2);
    std::vector<int> expected = {6, 12};
    if(result != expected){
        return false;  
    }return true;
} 

int main(){
    printf("%d\n", test_operation_to_info());
    printf("%d\n", test_operation_to_pooling_info());
    printf("%d\n", test_calc_reduction_layers());
    return 0;
}