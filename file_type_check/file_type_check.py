import os
import filetype

# 支持文件类型
# 用16进制字符串的目的是可以知道文件头是多少字节
# 各种文件头的长度不一样，少半2字符，长则8字符

# 支持文件类型
# 用16进制字符串的目的是可以知道文件头是多少字节
# 各种文件头的长度不一样，少半2字符，长则8字符


def typeList():
    print('获取文件格式十六进制码表……')
    return {
        "d0cf11e0a1b11ae10000": 'xls',
        '504b030414': 'docx_docx_xlsx',  # docx, pptx, xlsx
    }

# 字节码转16进制字符串


def bytes2hex(bytes):
    # print('关键码转码……');
    num = len(bytes)
    hexstr = u""
    for i in range(num):
        t = u"%x" % bytes[i]
        if len(t) % 2:
            hexstr += u"0"
        hexstr += t
    return hexstr.upper()


# 获取文件类型
def filetype_get(filename):
    # docx文件内容为空时，这里会返回
    if os.path.getsize(filename) == 0:
        print("文件内容为空")
        return "xxxx"

    print('读文件二进制码中……')
    binfile = open(filename, 'rb')  # 必需二制字读取
    print('读取20字节关键码……')
    bins = binfile.read(20)  # 提取20个字符
    binfile.close()  # 关闭文件流
    bins = bytes2hex(bins)  # 转码
    bins = bins.lower()  # 小写
    print(bins)
    tl = typeList()  # 文件类型
    ftype = 'unknown'
    print('关键码比对中……')
    for hcode in tl.keys():
        lens = len(hcode)  # 需要的长度
        if bins[0:lens] == hcode:
            ftype = tl[hcode]
            break
    if ftype == 'unknown':  # 全码未找到，优化处理，码表取5位验证
        bins = bins[0:5]
        for hcode in tl.keys():
            if len(hcode) > 5 and bins == hcode[0:5]:
                ftype = tl[hcode]
                break
    return ftype


def GuessFileType(pathfile):
    print('GuessFileType %s ...' % pathfile)

    kind = filetype.guess(pathfile)
    if kind is None:
        print('Cannot guess file type!')
        return

    print('File name: %s' % os.path.basename(pathfile))
    print('File extension: %s' % kind.extension)
    print('File MIME type: %s' % kind.mime)


if __name__ == '__main__':
    pathfile = r'D:\tmp\111.xlsx'
    # pathfile = r'D:\tmp\2222.doc'
    # pathfile = r'D:\tmp\3333.docx'
    # pathfile = r'D:\tmp\444.ppt'
    # pathfile = r'D:\tmp\666.pptx'
    # pathfile = r'D:\tmp\sss.txt'

    ftype = filetype_get(pathfile)
    print(ftype)
