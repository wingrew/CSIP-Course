### m5C检测工作流程整理
---

#### 1. 参考索引构建
- **软件**：hisat3n-build, samtools
- **推荐参数 (hisat3n-build)**：`-p 12 --base-change C,T`
- **功能描述**：构建参考基因组索引，适用于RNA亚硫酸盐测序（Bis-seq），通过`--base-change C,T`支持C到T转换的比对，`-p 12`指定12个线程以加速索引构建。

---

#### 2. 数据清洗
- **软件**：cutseq
- **推荐参数**：`-t 20 -A INLINE -m 20 --trim-polyA --ensure-inline-barcode`
- **功能描述**：修剪高通量测序数据中的接头序列，移除polyA尾（`--trim-polyA`），确保内联条形码存在（`--ensure-inline-barcode`），并设置最低质量阈值（`-t 20`）和最小序列长度（`-m 20`）以提高数据质量。

---

#### 3. rRNA、tRNA过滤与基因组比对
- **软件**：hisat3n, samtools
- **参数**：
  - **rRNA/tRNA比对 (hisat3n)**：`--base-change C,T --mp 8,2 --no-spliced-alignment --directional-mapping`
  - **基因组比对 (hisat3n)**：`--base-change C,T --pen-noncansplice 20 --mp 4,1 --directional-mapping`
- **功能描述**：
  - **rRNA/tRNA过滤**：通过`--no-spliced-alignment`禁用剪接比对，`--base-change C,T`支持C到T转换，`--mp 8,2`设置错配罚分，`--directional-mapping`启用方向性比对以过滤rRNA和tRNA序列。
  - **基因组比对**：将清洗后的读长比对到参考基因组，`--pen-noncansplice 20`提高非规范剪接罚分，`--mp 4,1`调整错配罚分，优化m5C检测的比对精度。

---

#### 4. 排序与去重
- **软件**：samtools sort, java + umicollapse.jar
- **推荐参数**：
  - **samtools**：`-@ 20 -m 3G --write-index`
  - **umicollapse**：`bam -t 2 -T 20 --data naive --merge avgqual --two-pass`
- **功能描述**：
  - **排序 (samtools sort)**：对BAM文件排序（`-@ 20`使用20线程，`-m 3G`分配3GB内存），生成索引（`--write-index`）以便后续分析。
  - **去重 (umicollapse)**：基于UMI（独特分子标识符）去除PCR重复（`-t 2 -T 20`设置阈值），`--data naive`处理原始数据，`--merge avgqual`以平均质量合并读长，`--two-pass`启用两遍去重算法以提高准确性。

---

#### 5. 位点调用与过滤
- **软件**：samtools view, hisat3n-table, bgzip
- **主要参数**：
  - **samtools view**：
    - 初始过滤：`-e "rlen<100000"`
    - 高级过滤：`-@ 10, -e "[XM]20 <= (qlen-sclen) && [Zf] <= 3 && 3[Zf] <= [Zf]+[Yf]"`
  - **hisat3n-table**：`-p N, -u/-m, --alignments, --ref..., --base-change C,T`
  - **bgzip**：（无具体参数，压缩输出文件）
- **功能描述**：
  - **samtools view**：过滤BAM文件，`rlen<100000`限制读长长度，高级过滤基于比对质量（如`[XM]`错配数、软剪切长度等）筛选高质量比对。
  - **hisat3n-table**：生成m5C位点统计表，`-p N`指定线程数，`-u/-m`选择唯一或多重比对，`--base-change C,T`支持C到T转换分析。
  - **bgzip**：压缩统计表，优化存储和传输。

---

#### 6. 脚本功能描述
以下脚本来自 [m5C-UBSseq GitHub](https://github.com/y9c/m5C-UBSseq)，用于整合和筛选m5C位点统计数据：

- **join_pileup.py**
  - **功能**：整合同一样本在不同条件下的统计信息。
  - **依赖包**：argparse, polars
  - **主要方法**：`pl.read_csv`（读取CSV数据），`pl.join`（合并数据集），`pl.fill_null`（填充空值）。
  
- **group_pileup.py**
  - **功能**：整合同一组内多个样本的统计信息，计算关键指标（如m5C覆盖率）。
  - **依赖包**：argparse, polars
  - **主要方法**：`pl.read_ipc`（读取IPC格式数据），`pl.join`（合并数据集），`pl.sum_horizontal`（水平求和）。

- **select_sites.py**
  - **功能**：对组统计结果进行初步筛选，过滤不符合阈值的位点，输出候选m5C位点。
  - **依赖包**：argparse, polars
  - **主要方法**：`pl.read_ipc`（读取数据），`pl.filter`（筛选数据），`pl.unique`（去重）。

- **filter_sites.py**
  - **功能**：计算未转换背景比例，进行显著性检验，筛选最终m5C候选位点。
  - **依赖包**：argparse, polars, scipy
  - **主要方法**：`pl.join`（合并数据），`pl.with_columns`（添加计算列），`binomtest`（二项分布显著性检验）。

---

此外，提供snakefile示例、snakemake_ssh及splitbam
