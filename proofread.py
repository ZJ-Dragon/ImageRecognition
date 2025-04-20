# 校对数据集的标签文件，检查是否有缺失的类别编号

import glob, os, collections

label_dir = "datasets/labels/train"   # 修改为你的 labels 路径
all_ids = []
for f in glob.glob(f"{label_dir}/**/*.txt", recursive=True):
    with open(f) as lab:
        for line in lab:
            if line.strip():
                cid = int(line.split()[0])
                all_ids.append(cid)

print("发现的类别编号：", sorted(set(all_ids)))
print("最大类别编号：", max(all_ids))