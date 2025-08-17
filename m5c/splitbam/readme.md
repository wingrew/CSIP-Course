splibam.cpp编译命令
`g++ -o m5c-UBSseq/splitbam splitbam.cpp -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib -lhts`

该程序作用：切分.bam文件

示例
./splitbam {input} -o {folder}" -@ {threads} -n {num} -r $(samtools idxstats {input}| awk '{{s+=$3}} END {{print s}}')

-o 输出文件夹
-n 切割份数
-r read总数
