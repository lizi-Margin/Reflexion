import subprocess
import sys
import time
import os

def run_multiple_games(num_runs=1):
    """
    运行指定次数的游戏
    
    Args:
        num_runs (int): 运行次数，默认15次
    """
    success_count = 0
    fail_count = 0
    
    for i in range(num_runs):
        print(f"\n{'='*50}")
        print(f"开始运行第 {i+1}/{num_runs} 次游戏")
        print(f"{'='*50}")
        
        try:
            # 设置环境变量以确保正确的编码
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            # 运行 run.py 脚本，不捕获输出以避免编码问题
            result = subprocess.run([sys.executable, "run.py"], 
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
    print(f"{'='*50}")

if __name__ == "__main__":
    # 可以通过命令行参数指定运行次数，否则默认15次
    if len(sys.argv) > 1:
        try:
            num_runs = int(sys.argv[1])
        except ValueError:
            print("参数必须是整数，使用默认值15")
            num_runs = 3
    else:
        num_runs = 3
    
    run_multiple_games(num_runs)