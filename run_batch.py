import subprocess
import sys
import time
import os

def run_multiple_games(script_name, num_runs=15):
    """
    运行指定次数的游戏
    
    Args:
        script_name (str): 要运行的Python脚本文件名
        num_runs (int): 运行次数，默认15次
    """
    success_count = 0
    fail_count = 0
    
    # 检查脚本文件是否存在
    if not os.path.exists(script_name):
        print(f"错误: 脚本文件 '{script_name}' 不存在!")
        return
    
    print(f"将运行脚本: {script_name}")
    print(f"运行次数: {num_runs}")
    
    for i in range(num_runs):
        print(f"\n{'='*50}")
        print(f"开始运行第 {i+1}/{num_runs} 次游戏")
        print(f"{'='*50}")
        
        try:
            # 设置环境变量以确保正确的编码
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            # 运行指定的脚本，不捕获输出以避免编码问题
            result = subprocess.run([sys.executable, script_name], 
                                  env=env,
                                  timeout=3000)  # 5分钟超时
            
            if result.returncode == 0:
                print(f"第 {i+1} 次游戏运行成功!")
                success_count += 1
            else:
                print(f"第 {i+1} 次游戏运行失败!")
                fail_count += 1
                
        except subprocess.TimeoutExpired:
            print(f"第 {i+1} 次游戏运行超时!")
            fail_count += 1
            
        except Exception as e:
            print(f"第 {i+1} 次游戏运行出现异常: {e}")
            fail_count += 1
        
        # 每次运行之间稍作间隔
        if i < num_runs - 1:  # 最后一次不需要等待
            print(f"等待5秒后开始下次游戏...")
            time.sleep(5)
    
    # 输出统计结果
    print(f"\n{'='*50}")
    print(f"游戏运行完成统计:")
    print(f"成功次数: {success_count}")
    print(f"失败次数: {fail_count}")
    print(f"总运行次数: {num_runs}")
    print(f"成功率: {success_count/num_runs*100:.2f}%")
    print(f"{'='*50}")

if __name__ == "__main__":
    # 命令行参数格式: python run_batch.py xxx.py 8
    if len(sys.argv) < 2:
        print("使用方法: python run_batch.py <脚本文件名> [运行次数]")
        print("示例: python run_batch.py run.py 8")
        sys.exit(1)
    
    script_name = sys.argv[1]
    
    # 获取运行次数参数
    if len(sys.argv) > 2:
        try:
            num_runs = int(sys.argv[2])
        except ValueError:
            print("错误: 运行次数必须是整数!")
            sys.exit(1)
    else:
        num_runs = 15  # 默认值
    
    run_multiple_games(script_name, num_runs)