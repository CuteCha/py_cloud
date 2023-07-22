def stat_parameter(num_token, seq_len, emb_dim, h_dim, v_dim, num_head, f_dim, num_block):
    emb_parameter = num_token * emb_dim + seq_len * emb_dim
    qkv_parameter = (emb_dim * h_dim + h_dim) + (emb_dim * h_dim + h_dim) + (emb_dim * v_dim + v_dim)
    multi_head_parameter = qkv_parameter * num_head + (num_head * v_dim) * emb_dim
    ffn_parameter = (emb_dim * f_dim + f_dim) + (f_dim * emb_dim + emb_dim)
    ln_parameter = 4 * emb_dim
    cls_parameter = emb_dim * num_token + num_token

    pre_model_parameter = emb_parameter + (multi_head_parameter + ffn_parameter + ln_parameter) * num_block
    total_parameter = pre_model_parameter + cls_parameter

    print(f"emb_parameter: {format(emb_parameter / pre_model_parameter, '.2%')}, {format(emb_parameter, ',.1f')}\n"
          f"multi_head_parameter: {format(multi_head_parameter * num_block / pre_model_parameter, '.2%')}, "
          f"{format(multi_head_parameter * num_block, ',.1f')}, {multi_head_parameter}\n"
          f"ffn_parameter: {format(ffn_parameter * num_block / pre_model_parameter, '.2%')}, "
          f"{format(ffn_parameter * num_block, ',.1f')}, {ffn_parameter}\n"
          f"ln_parameter: {format(ln_parameter * num_block / pre_model_parameter, '.2%')}, "
          f"{format(ln_parameter * num_block, ',.1f')}, {ln_parameter}\n"
          f"cls_parameter: {format(cls_parameter / pre_model_parameter, '.2%')}, "
          f"{format(cls_parameter, ',.1f')}, {cls_parameter}\n"
          f"pre_model_parameter: {format(pre_model_parameter, ',.1f')}\n"
          f"total_parameter: {format(total_parameter, ',.2f')}")


def stat_ops(num_token, batch_size, seq_len, emb_dim, h_dim, v_dim, num_head, f_dim, num_block, total_token):
    qkv_ops = 2 * 2 * batch_size * seq_len * emb_dim * h_dim + 2 * batch_size * seq_len * emb_dim * v_dim
    multi_head_ops = qkv_ops * num_head + 2 * batch_size * seq_len * v_dim * emb_dim
    ffn_ops = 2 * batch_size * seq_len * emb_dim * f_dim + 2 * batch_size * seq_len * f_dim * emb_dim
    cls_ops = 2 * batch_size * seq_len * emb_dim * num_token

    pre_model_ops = (multi_head_ops + ffn_ops) * num_block
    one_step_ops = pre_model_ops + cls_ops
    one_step_ops = num_block * (24 * batch_size * seq_len * emb_dim ** 2 + 4 * batch_size * seq_len ** 2 * emb_dim) \
                   + 2 * batch_size * seq_len * emb_dim * num_token
    total_ops = 3 * one_step_ops * total_token / (batch_size * seq_len)

    print("-" * 72 +
          f"\npre_model_ops: {format(pre_model_ops)}\n"
          f"one step ops: {format(one_step_ops)}\n"
          f"total ops: {total_ops}")


def main():
    # num_token = 50257
    # seq_len = 1024
    # num_head = 8
    # emb_dim = 4096  # 8192
    # h_dim = 768 * num_head
    # v_dim = 768
    # f_dim = 512 * emb_dim
    # num_block = 48

    # bert
    num_token = 30522
    seq_len = 512
    num_head = 12
    emb_dim = 768
    h_dim = emb_dim / num_head
    v_dim = emb_dim / num_head
    f_dim = 4 * emb_dim
    num_block = 12

    # GPT
    num_token = 50257
    batch_size = 32768
    seq_len = 2048
    num_head = 96
    emb_dim = 12288
    h_dim = emb_dim / num_head
    v_dim = emb_dim / num_head
    f_dim = 4 * emb_dim
    num_block = 96
    total_token = 300 * 1e9

    stat_parameter(num_token, seq_len, emb_dim, h_dim, v_dim, num_head, f_dim, num_block)
    stat_ops(num_token, batch_size, seq_len, emb_dim, h_dim, v_dim, num_head, f_dim, num_block, total_token)


if __name__ == '__main__':
    main()
