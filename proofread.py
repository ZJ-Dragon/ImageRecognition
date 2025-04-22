# 校对数据集的标签文件，检查是否有缺失的类别编号

import glob, os, collections

label_root = "/Volumes/MobileWorkstation/Projects/AutoPilot/datasets/face03.250422.0006/images"   # 指向 labels 总目录
txt_files  = glob.glob(f"{label_root}/**/*.txt", recursive=True)

print(f"共找到 {len(txt_files)} 个 .txt 标注文件")
if not txt_files:
    print("⚠️ 路径不对？或没有任何标签文件。")
    quit()

all_ids = []
for f in txt_files:
    with open(f) as lab:
        for line in lab:
            parts = line.strip().split()
            if len(parts) >= 1:
                try:
                    cid = int(float(parts[0]))   # 第一个字段是类别编号
                    all_ids.append(cid)
                except ValueError:
                    print(f"⚠️ 非法标签行 {line.strip()} in {f}")

if all_ids:
    print("出现的类别编号：", sorted(set(all_ids)))
    print("最大类别编号：", max(all_ids))
else:
    print("⚠️ 所有 .txt 里都没有合法的类别编号！请检查标签文件内容。")