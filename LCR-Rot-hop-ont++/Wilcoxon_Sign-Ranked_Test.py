import pandas as pd
from scipy.stats import wilcoxon

def perform_wilcoxon_test(file1, file2):
    # Read data from CSV files
    data1 = pd.read_csv(file1, header = None)
    data2 = pd.read_csv(file2, header = None)
    
    # Check if the datasets have the same length
    if len(data1) != len(data2):
        raise ValueError("Datasets must have the same length for a paired test")
    
    list1 = data1.values.flatten().tolist()
    list2 = data2.values.flatten().tolist()
    
    # Perform Wilcoxon signed-rank test
    stat, p_value = wilcoxon(list1, list2)
    
    return stat, p_value

if __name__ == "__main__":
    # Example usage:
    file1_path = "./data/bootstrap_results/2015_test.csv"
    file2_path = "./data/bootstrap_results/2015_test_wparam.csv"
    
    try:
        stat, p_value = perform_wilcoxon_test(file1_path, file2_path)
        print("Wilcoxon Signed-Rank Test Results for: ", file1_path, " - " ,file2_path)
        print(f"Statistic: {stat}")
        print(f"P-value: {p_value}")
        if p_value < 0.05:
            print("Conclusion: Reject the null hypothesis")
        else:
            print("Conclusion: Fail to reject the null hypothesis")
    except Exception as e:
        print("Error:", e)