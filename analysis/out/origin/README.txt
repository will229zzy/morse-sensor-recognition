Origin 画图数据说明
================================
总样本 2661,正确 2659,总准确率 99.92%
识别错误(真->判,次数): [('I', 'A', 1), ('I', 'N', 1)]

1) 面板(c) 波形叠加图  ——  waveforms/<字母>.csv
   - 第 1 列 time_s = 时间(秒);其余每列 rep_XX = 一次重复的 ΔR/R0 (%)
   - Origin: 导入某字母的 csv -> 选中 time_s 作 X、所有 rep_* 作 Y -> Plot: Line
   - 26 个字母各画一个小图,排成 2x13,即得面板(c)

2) 面板(e) 混淆矩阵  ——  confusion_matrix.csv(计数) / confusion_matrix_percent.csv(每行百分比)
   - 行 = 真实字母,列 = 预测字母
   - Origin: 导入 -> 设为矩阵(Matrix)-> Plot: Heatmap/Image;
     想让少量错误可见,可用对数色标,或在对应格子单独标注数值
   - 标题写 Accuracy: 99.9%

3) 逐字母准确率  ——  per_letter_accuracy.csv
   - letter / n_samples / accuracy_%  -> Origin 条形图

坐标轴:纵轴 ΔR/R0 (%)(相对电阻变化),横轴 时间 (s),与参考论文 Fig.5 一致。
