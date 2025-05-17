import os
import subprocess

# 수정: 여기에 최상위 폴더 경로 설정
base_folder = r"C:\Users\kkjoo\Downloads\Fashion\Validation\원천데이터"

# 하위 폴더 리스트 자동 탐색
subfolders = [os.path.join(base_folder, name) for name in os.listdir(base_folder)
              if os.path.isdir(os.path.join(base_folder, name))]

# 실행할 Python 인터프리터 경로
python_exec = "python"

print("[INFO] Launching parallel processes for each subfolder...\n")

processes = []
for folder in subfolders:
    print(f"[LAUNCH] {folder}")
    p = subprocess.Popen([python_exec, "person_autocrop_worker.py", folder])
    processes.append(p)

# 모든 프로세스 완료 대기
for p in processes:
    p.wait()

print("[ALL DONE] All folders processed.")
