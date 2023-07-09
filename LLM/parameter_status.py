import numpy as np


def stat_parameter(num_token, seq_len, emb_dim, h_dim, v_dim, num_head, f_dim, num_block):
    emb_parameter = num_token * emb_dim + seq_len * emb_dim
    qkv_parameter = (emb_dim * h_dim + h_dim) + (emb_dim * h_dim + h_dim) + (emb_dim * v_dim + v_dim)
    multi_head_parameter = qkv_parameter * num_head + (num_head * v_dim) * emb_dim
    ffn_parameter = (emb_dim * f_dim + f_dim) + (f_dim * emb_dim + emb_dim)
    ln_parameter = 4 * emb_dim
    cls_parameter = emb_dim * num_token + num_token

    total_parameter = emb_parameter + (multi_head_parameter + ffn_parameter + ln_parameter) * num_block + cls_parameter

    print(f"emb_parameter: {emb_parameter}\nmulti_head_parameter: {multi_head_parameter}\n"
          f"ffn_parameter: {ffn_parameter}\nln_parameter: {ln_parameter}\n"
          f"cls_parameter: {cls_parameter}\ntotal_parameter: {total_parameter}")

    print("-" * 36)
    print(f"emb_parameter: {100 * emb_parameter / total_parameter}\n"
          f"multi_head_parameter: {100 * num_block * multi_head_parameter / total_parameter}\n"
          f"ffn_parameter: {100 * num_block * ffn_parameter / total_parameter}\n"
          f"ln_parameter: {100 * num_block * ln_parameter / total_parameter}\n"
          f"cls_parameter: {100 * cls_parameter / total_parameter}")


def main():
    num_token = 50257
    seq_len = 1024
    num_head = 8
    emb_dim = 4096  # 8192
    h_dim = 768 * num_head
    v_dim = 768
    f_dim = 512 * emb_dim
    num_block = 48

    stat_parameter(num_token, seq_len, emb_dim, h_dim, v_dim, num_head, f_dim, num_block)


if __name__ == '__main__':
    main()
