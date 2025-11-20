# import re
from graph_net import analysis_util
    
def main():
    tolerance_list =[-3,-2,-1,1,2]
    # your log file path here
    log_file_path = "/work/.BCloud/log_20251013_175058_torch_inductor_full.log"
    for tolerance in tolerance_list:
        print("tolerance:",tolerance)
        for i in analysis_util.get_incorrect_models(tolerance, log_file_path):
            print(i)
    
if __name__ == "__main__":
    main()
