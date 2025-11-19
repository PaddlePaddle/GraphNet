# import re
from graph_net import analysis_util

def get_incorrect_models(tolerance,log_file_path)->list:
    """
    根据给定的 tolerance 和日志文件，定位存在精度问题的模型。
    
    参数:
        tolerance (float): 容忍度，精度要求。
        log_file_path (str): 日志文件路径，包含模型的日志信息。
        
    返回:
        List[str]: 包含存在精度问题的模型路径的字符串数组。
    """
    failed_models = []
    list = analysis_util.parse_logs_to_data(log_file_path)

    for i in list:
        isok,str1 = analysis_util.check_sample_correctness(i,tolerance)
        # print(isok,str1)
        if not isok:
            failed_models.append(i.get('configuration').get('model'))
    return failed_models
    
def main():
    tolerance_list =[-3,-2,-1,1,2]
    log_file_path = "/work/.BCloud/work/GraphNet/graph_net/log_20251013_175058_torch_inductor_full.log"
    # log_file_path = "/work/.BCloud/work/GraphNet/graph_net/log_test_target_device-xpu_p800_nope-pd_20251111_2.txt"
    for tolerance in tolerance_list:
        print("tolerance:",tolerance)
        for i in get_incorrect_models(tolerance, log_file_path):
            print(i)
    
if __name__ == "__main__":
    main()
