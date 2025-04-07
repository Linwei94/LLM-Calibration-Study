import os
import json
import csv
import re

def process_results_to_csv(results_folder='results', output_csv='output.csv'):
    rows = []
    for filename in os.listdir(results_folder):
        if filename.endswith('.json'):
            filepath = os.path.join(results_folder, filename)
            # Extract benchmark, model, confidence_mode, num_samples from filename
            benchmark = filename.split('_')[0]
            model_name = filename.split('_')[1]
            conf_mode = "verbal_cot" if "verbal_cot" in filename else "verbal"
            num_samples = filename.split('_')[-1].replace('.json', '')
            num_samples = num_samples if num_samples.isdigit() else '0'
            if "llama" in model_name:
                model_family = "LLaMA"
                if "8b" in model_name:
                    model_size = 8
                elif "70b" in model_name:
                    model_size = 70
                elif "405b" in model_name:
                    model_size = 405
                elif "maverick" in model_name:
                    model_size = 400
            elif "gpt" in model_name:
                model_family = "GPT"
                if "4o-mini" in model_name:
                    model_size = 8
                elif "4o" in model_name:
                    model_size = 200
            # Load JSON content
            with open(filepath, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON: {filename}")
                    continue
            
            # 加载 JSON 内容（单个字典）
            with open(filepath, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"无法解析 JSON 文件: {filename}")
                    continue

            # 构造一行数据
            row = {
                'benchmark': benchmark,
                'model': model_name,
                'conf_mode': conf_mode,
                'n_samples': int(num_samples),
                'model_family': model_family,
                'model_size': model_size,
                'score': data.get('score', None),
                'verbal_ece': data.get('verbal_ece', None)
            }
            rows.append(row)

    # Write all to CSV
    output_csv_file_path = os.path.join("plots", output_csv)
    if rows:
        os.makedirs(os.path.dirname(output_csv_file_path), exist_ok=True)  # Create directory if needed
        with open(output_csv_file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved to {output_csv_file_path}")
    else:
        print("No data found.")

if __name__ == "__main__":
    process_results_to_csv()