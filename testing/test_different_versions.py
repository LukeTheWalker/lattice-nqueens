import os
import subprocess

def run_process(command):
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout.strip()

def compare_outputs(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        output1 = f1.read().strip()
        output2 = f2.read().strip()
        return output1, output2

def main():
    test_cases_dir = 'test_cases'
    process1_command = ['./version1']
    process2_command = ['./version2']

    for test_file in os.listdir(test_cases_dir):
        test_file_path = os.path.join(test_cases_dir, test_file)
        
        run_process(process1_command + [test_file_path])
        run_process(process2_command + [test_file_path])
        
        output1_file = f'version1.txt'
        output2_file = f'version2.txt'

        output1, output2 = compare_outputs(output1_file, output2_file)
        
        if not output1 == output2:
            print(f'Error: Outputs differ for test case {test_file}')
            print(f'Check files: {output1_file} and {output2_file}')
            print(f'Output1: {output1}')
            print(f'Output2: {output2}')
            input('Press Enter to continue...')

if __name__ == '__main__':
    main()