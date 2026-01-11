"""
主程序：Wordle玩家表现预测完整流程
运行此文件将依次执行所有步骤
"""

import os
import sys
import subprocess
import time

def run_step(step_file, step_name):
    """运行单个步骤"""
    print("\n" + "="*70)
    print(f"执行步骤: {step_name}")
    print("="*70)
    
    if not os.path.exists(step_file):
        print(f"错误: 未找到文件 {step_file}")
        return False
    
    try:
        result = subprocess.run([sys.executable, step_file], 
                              capture_output=False,
                              text=True,
                              encoding='utf-8')
        if result.returncode == 0:
            print(f"\n✓ {step_name} 完成")
            return True
        else:
            print(f"\n✗ {step_name} 失败 (退出码: {result.returncode})")
            return False
    except Exception as e:
        print(f"\n✗ {step_name} 执行出错: {e}")
        return False

def main():
    """主函数"""
    print("="*70)
    print("Wordle玩家表现预测 - 完整流程")
    print("="*70)
    print("\n本程序将依次执行以下步骤:")
    print("  1. 数据清洗")
    print("  2. 特征工程")
    print("  3. 构建模型")
    print("  4. 训练模型")
    print("  5. 评估模型")
    print("  6. 可视化结果")
    print("\n提示: 如果某个步骤已经完成，可以跳过该步骤直接运行后面的步骤")
    
    # 询问是否继续
    response = input("\n是否开始执行? (y/n): ").strip().lower()
    if response != 'y':
        print("已取消")
        return
    
    # 定义所有步骤
    steps = [
        ('step1_data_cleaning.py', '步骤1: 数据清洗'),
        ('step2_feature_engineering.py', '步骤2: 特征工程'),
        ('step3_build_model.py', '步骤3: 构建模型'),
        ('step4_train_model.py', '步骤4: 训练模型 '),
        ('step5_evaluate_model.py', '步骤5: 评估模型'),
        ('step6_visualization.py', '步骤6: 可视化结果'),
    ]
    
    # 执行每个步骤
    start_time = time.time()
    success_count = 0
    
    for step_file, step_name in steps:
        if run_step(step_file, step_name):
            success_count += 1
        else:
            print(f"\n错误: {step_name} 执行失败，流程中断")
            print("请检查错误信息并修复后重新运行")
            break
        
        # 步骤间暂停
        time.sleep(1)
    
    # 总结
    elapsed_time = time.time() - start_time
    print("\n" + "="*70)
    print("流程执行完成")
    print("="*70)
    print(f"成功步骤: {success_count}/{len(steps)}")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    
    if success_count == len(steps):
        print("\n✓ 所有步骤执行成功!")
        print("\n生成的文件:")
        print("  - data_cleaned.csv: 清洗后的数据")
        print("  - X_train.npy, X_test.npy: 训练和测试特征")
        print("  - y_train.npy, y_test.npy: 训练和测试标签")
        print("  - model_lstm.pth: 训练好的模型")
        print("  - evaluation_results.json: 评估结果")
        print("  - visualizations/: 可视化图表目录")
        print("\n下一步: 查看生成的报告和可视化结果")
    else:
        print(f"\n部分步骤失败，请检查错误信息")

if __name__ == '__main__':
    main()
