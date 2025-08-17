#include <htslib/hts.h>
#include <htslib/sam.h>
#include <htslib/thread_pool.h>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <getopt.h>
#include <ctime>

// 输出模式，默认为 BAM
char out_mode[5] = "wb";

// 输出格式结构体
htsFormat *output_format = (htsFormat *)calloc(1, sizeof(htsFormat));

// 打印帮助信息
int usage(FILE *fp, int is_long_help) {
    fprintf(fp,
            "\n"
            "Usage: ./splitbam [options] <in.bam>|<in.sam>|<in.cram> \n"
            "\n"
            "Options:\n"
            // 输出选项
            "  -b       output BAM\n"
            "  -C       output CRAM (requires -T)\n"
            "  -o FILE  output file name \n"
            "  -T FILE  reference in the fasta format (required for reading and writing CRAMs)\n"
            "  -@ INT   Number of threads to use\n"
            "  -q INT   Mapping quality filter\n"
            "  -m       Discard unmapped reads (default off)\n"
            "  -n INT   Number of output file \n"
            "  -v       Verbose mode\n");
    fprintf(fp,
            "\nNotes:\n"
            "\n"
            "1. This program is useful for splitting a sorted BAM/CRAM into multiple files\n");

    return 0;
}

// 解析和处理测序数据
void parse_sequencingdata(char *fn_out, char *fname, int nthreads, int reads_per_file) {
    htsThreadPool p = {NULL, 0};
    samFile *in = NULL;
    samFile *out = NULL;
    char output_filename[2048] = "";
    int file_index = 0;
    char *last_read_name = NULL;

    // 打开输入文件
    if ((in = sam_open_format(fname, "r", output_format)) == NULL) {
        fprintf(stderr, "[%s] Error: Cannot open file: %s\n", __FUNCTION__, fname);
        exit(1);
    }

    // 读取文件头
    bam_hdr_t *hdr = sam_hdr_read(in);

    // 初始化线程池
    if (nthreads > 1) {
        if (!(p.pool = hts_tpool_init(nthreads))) {
            fprintf(stderr, "Error: Failed to create thread pool\n");
            exit(1);
        }
        hts_set_opt(in, HTS_OPT_THREAD_POOL, &p);
    }

    // 初始化 BAM 记录
    bam1_t *b = bam_init1();

    // 批量写入的缓冲区
    #define BATCH_SIZE 1000
    bam1_t *batch[BATCH_SIZE];
    int batch_count = 0;

    int ret;
    int total_reads = 0;
    int reads_in_file = 0;

    // 读取并处理每条记录
    while ((ret = sam_read1(in, hdr, b)) >= 0) {
        // 如果当前文件中的记录数超过限制，关闭当前文件并打开新文件
        if (reads_in_file >= reads_per_file) {
            char *current_read_name = bam_get_qname(b);
            if (last_read_name == NULL || strcmp(last_read_name, current_read_name) != 0) {
                if (out) {
                    // 写入剩余的记录
                    for (int i = 0; i < batch_count; i++) {
                        if (sam_write1(out, hdr, batch[i]) < 0) {
                            fprintf(stderr, "Error: Failed to write record to output file\n");
                            exit(1);
                        }
                        bam_destroy1(batch[i]);
                    }
                    batch_count = 0;
                    if (sam_close(out) != 0) {
                        fprintf(stderr, "Error: Failed to close output file\n");
                        exit(1);
                    }
                    out = NULL;
                }
                reads_in_file = 0;
            }
            if (last_read_name) free(last_read_name);
            last_read_name = strdup(current_read_name);
        }

        // 打开新文件
        if (out == NULL) {
            if (out_mode[1] == 'b') {
                snprintf(output_filename, sizeof(output_filename), "%s_%d.bam", fn_out, file_index++);
            } else {
                snprintf(output_filename, sizeof(output_filename), "%s_%d.cram", fn_out, file_index++);
            }

            if ((out = sam_open_format(output_filename, out_mode, output_format)) == NULL) {
                fprintf(stderr, "Error: Cannot open output file: %s\n", output_filename);
                exit(1);
            }

            if (nthreads > 1 && out) {
                hts_set_opt(out, HTS_OPT_THREAD_POOL, &p);
            }

            if (sam_hdr_write(out, hdr) != 0) {
                fprintf(stderr, "Error: Failed to write header to output file\n");
                exit(1);
            }
        }

        // 将记录添加到批量写入缓冲区
        batch[batch_count++] = bam_dup1(b);
        if (batch_count == BATCH_SIZE) {
            for (int i = 0; i < batch_count; i++) {
                if (sam_write1(out, hdr, batch[i]) < 0) {
                    fprintf(stderr, "Error: Failed to write record to output file\n");
                    exit(1);
                }
                bam_destroy1(batch[i]);
            }
            batch_count = 0;
        }
        reads_in_file++;
        total_reads++;
    }

    // 写入剩余的记录
    for (int i = 0; i < batch_count; i++) {
        if (sam_write1(out, hdr, batch[i]) < 0) {
            fprintf(stderr, "Error: Failed to write record to output file\n");
            exit(1);
        }
        bam_destroy1(batch[i]);
    }

    // 关闭文件和释放资源
    if (out && sam_close(out) != 0) {
        fprintf(stderr, "Error: Failed to close output file\n");
        exit(1);
    }
    if (sam_close(in) != 0) {
        fprintf(stderr, "Error: Failed to close input file\n");
        exit(1);
    }

    bam_destroy1(b);
    if (last_read_name) free(last_read_name);
    hts_opt_free((hts_opt *)output_format->specific);
    free(output_format);

    if (nthreads > 1) {
        hts_tpool_destroy(p.pool);
    }

    fprintf(stderr, "Processed %d reads in total\n", total_reads);
}

int main(int argc, char **argv) {
    clock_t t = clock();
    time_t t2 = time(NULL);

    char *fname = NULL;
    char *fn_out = NULL;
    int c;
    int nthreads = 1;
    long long int reads_per_file = 5000000; // 每个文件包含 500 万条记录
    int num = 1;
    if (argc == 1) {
        usage(stdout, 0);
        return 0;
    }

    // 解析命令行参数
    static struct option lopts[] = {
        {"add", 1, 0, 0},
        {"append", 0, 0, 0},
        {"delete", 1, 0, 0},
        {"verbose", 0, 0, 0},
        {"create", 1, 0, 'c'},
        {"file", 1, 0, 0},
        {NULL, 0, NULL, 0}};
    
    while ((c = getopt_long(argc, argv, "bCo:T:@:q:m:n:p:r:v", lopts, NULL)) >= 0) {
        switch (c) {
        case 'o':
            fn_out = strdup(optarg);
            break;
        case '@':
            nthreads = atoi(optarg);
            break;
        case 'r':
            reads_per_file = atoi(optarg) / num + 20;
            break;
        case 'n':
            num = atoi(optarg);
            break;
        case 'v':
            // 未实现详细模式
            break;
        case '?':
            if (optopt == '?') {
                return usage(stdout, 0);
            } else {
                fprintf(stderr, "Error: Invalid option -- '%c'\n", optopt);
                return 1;
            }
        default:
            fname = strdup(optarg);
            break;
        }
    }
    
    if (optind < argc) {
        fname = strdup(argv[optind]);
    }

    // 检查输入文件和输出文件是否指定
    if (!fname) {
        fprintf(stderr, "Error: No input file specified\n");
        usage(stdout, 0);
        return 1;
    }

    if (!fn_out) {
        fprintf(stderr, "Error: No output file specified\n");
        usage(stdout, 0);
        return 1;
    }
    
    // 处理测序数据
    parse_sequencingdata(fn_out, fname, nthreads, reads_per_file);

    // 释放内存
    if (fname) free(fname);
    if (fn_out) free(fn_out);

    // 打印运行时间
    fprintf(stderr,
            "\t[ALL done] CPU time used = %.2f sec\n"
            "\t[ALL done] Wall time used = %.2f sec\n",
            (float)(clock() - t) / CLOCKS_PER_SEC, (float)(time(NULL) - t2));

    return 0;
}