import os
import subprocess
from multiprocessing import Process

SCRIPTS_DIR = 'scripts'


def run_script(script_name):
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    print(f"\nRunning {script_name}...\n")
    result = subprocess.run(['python', script_path], capture_output=True, text=True, encoding='utf-8')
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error running {script_name}: {result.stderr}")


def run_in_parallel(scripts):
    processes = []
    for script_name in scripts:
        process = Process(target=run_script, args=(script_name,))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


def main():
    run_script('data_preparation.py')
    scripts_to_parallelize = ['model_svm.py', 'model_lightgbm.py', 'model_lstm.py', 'model_cnn_lstm.py']
    run_in_parallel(scripts_to_parallelize)
    run_script('model_evaluation.py')
    run_script('results_analysis.py')

    print("\nAll scripts ran successfully!")


if __name__ == '__main__':
    main()
