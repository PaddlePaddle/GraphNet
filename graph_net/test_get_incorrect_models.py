# import re
from graph_net import analysis_util
    
def main():
    tolerance_list =[-3,-2,-1,1,2]
    log_file_path = "/work/.BCloud/clone/GraphNet/graph_net/log_20251013_175058_torch_inductor_full.log"
    # log_file_path = "/work/.BCloud/clone/GraphNet/graph_net/log_test_target_device-xpu_p800_nope-pd_20251111_2.txt"
    for tolerance in tolerance_list:
        print("tolerance:",tolerance)
        for i in analysis_util.get_incorrect_models(tolerance, log_file_path):
            print(i)
    
if __name__ == "__main__":
    main()
