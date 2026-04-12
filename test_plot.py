import sys
import os
import matplotlib
# 强行设置 Agg
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    print("="*50)
    print("1. Python 版本:", sys.version)
    print("2. Matplotlib 后端:", matplotlib.get_backend())
    print("3. 当前工作目录:", os.getcwd())
    
    try:
        print("4. 正在尝试在内存中生成简单图表...")
        plt.figure(figsize=(4, 4))
        plt.plot([1, 2, 3], [4, 5, 2])
        plt.title("Test Plot")
        
        save_path = os.path.join(os.getcwd(), 'test_output_image.png')
        print(f"5. 准备写入文件: {save_path}")
        
        # 这一步是生与死的关键
        plt.savefig(save_path)
        print("6. [完美成功] 图片已成功写入硬盘！")
        
    except Exception as e:
        print("\n[致命报错] 捕获到异常:")
        print(e)
    finally:
        print("="*50)

if __name__ == "__main__":
    main()