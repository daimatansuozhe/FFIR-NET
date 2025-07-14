import pandas as pd
import os
import shutil


def extract_top_features_and_process_stock_files(feature_ranking_path, stock_files_folder, num):
    """
    从特征排名文件中提取前n个特征，并处理股票文件保留这些特征和date列，保存到新文件夹。

    :param feature_ranking_path: 特征排名文件的路径
    :param stock_files_folder: 存放股票文件的文件夹路径
    :param num: 提取的特征数量
    """
    # 读取特征排名文件
    try:
        feature_ranking_df = pd.read_csv(feature_ranking_path)
    except FileNotFoundError:
        print(f"未找到特征排名文件 {feature_ranking_path}")
        return

    # 提取前n个特征
    top_n_features = feature_ranking_df['feature'].head(num).tolist()

    # 创建新文件夹（如果不存在）
    output_folder = f"{os.path.dirname(stock_files_folder)}/price_with_{num}_indicators"
    os.makedirs(output_folder, exist_ok=True)

    # 遍历所有股票文件
    for root, dirs, files in os.walk(stock_files_folder):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                try:
                    # 读取股票文件
                    stock_df = pd.read_csv(file_path)
                except FileNotFoundError:
                    print(f"未找到股票文件 {file_path}")
                    continue

                # 保留指定特征和date列
                columns_to_keep = ['date'] + [col for col in stock_df.columns if col in top_n_features]

                # 确保date列存在
                if 'date' not in stock_df.columns:
                    print(f"警告: 文件 {file} 中未找到date列")
                    columns_to_keep = columns_to_keep[1:]  # 移除'date'列（如果不存在）

                if not columns_to_keep:
                    print(f"警告: 文件 {file} 中未找到任何匹配的特征")
                    continue

                processed_df = stock_df[columns_to_keep]

                # 构建输出文件路径（保持原文件名）
                output_file_path = os.path.join(output_folder, file)

                # 保存处理后的文件
                processed_df.to_csv(output_file_path, index=False)
                print(f"已处理文件: {file_path} -> {output_file_path}")


if __name__ == "__main__":
    # 定义特征排名文件路径
    feature_ranking_file_path = 'feature_ranking_with_scores.csv'
    # 定义原始股票文件文件夹路径
    stock_files_folder_path = '../data/price/price_with_all_indicators'
    # 提取的特征数量
    num = 10

    extract_top_features_and_process_stock_files(feature_ranking_file_path, stock_files_folder_path, num)