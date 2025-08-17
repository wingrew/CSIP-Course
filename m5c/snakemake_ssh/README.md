命令执行：

1. `./snakemake_ssh/server.py -c ./snakemake_ssh/server.py.configs.yaml.default`
2. `snakemake --executor cluster-generic --jobs 999 --cluster-generic-submit-cmd "snakemake_ssh/ssh_parallel.py -i"`


你可以修改server.py.configs.yaml.default文件，来配置集群资源