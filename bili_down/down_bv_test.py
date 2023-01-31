# -*- coding: utf-8 -*-
import requests
import json
import time
import re
import urllib
import os
from urllib.request import urlretrieve, build_opener, install_opener


def get_by_bv(bv):
    base_url = "https://api.bilibili.com/x/web-interface/view?bvid="
    response = requests.request("GET", base_url + bv)
    print(response.status_code)
    print(response.content.decode("utf-8"))


def down_by_aid_cid(aid, cid):
    url = "https://api.bilibili.com/x/player/playurl?avid=" + aid + "&cid=" + cid + "&qn=80&type=mp4&platform=html5&high_quality=1"
    print(url)
    response = requests.request("GET", url)
    print(response.status_code)
    content = response.content.decode("utf-8")
    print(content)
    content_json = json.loads(content)
    d_url = content_json["data"]["durl"][0]["url"]
    print(d_url)

    print("start download ......")
    opener = build_opener()
    opener.addheaders = [('Origin', 'https://www.bilibili.com'),
                         ('Referer', "https://www.bilibili.com/video/"),
                         ('User-Agent',
                          'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36')]
    install_opener(opener)
    urlretrieve(url, "./xxx.mp4")
    video_data = requests.request("GET", d_url).content
    with open("./test.mp4", mode='wb') as fr:
        fr.write(video_data)
        print("download finished ......")


# 一般视频是mp4，音频是mp3
def down_file(file_url, file_type):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.66 Safari/537.36",
        "referer": "https://www.bilibili.com"
    }
    resp = requests.get(url=file_url, headers=headers)
    print(resp.status_code)

    title = re.findall(r'<h1 title="(.*?)" class="video-title">', resp.text)[0]
    print(f'文件名称：{title}')
    # 设置单次写入数据的块大小
    chunk_size = 1024
    # 获取文件大小
    file_size = int(resp.headers['content-length'])
    # 用于记录已经下载的文件大小
    done_size = 0
    # 将文件大小转化为MB
    file_size_MB = file_size / 1024 / 1024
    print(f'文件大小：{file_size_MB:0.2f} MB')
    start_time = time.time()
    with open(title + '.' + file_type, mode='wb') as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            done_size += len(chunk)
            print(f'\r下载进度：{done_size / file_size * 100:0.2f}%', end='')
    end_time = time.time()
    cost_time = end_time - start_time
    print(f'\n累计耗时：{cost_time:0.2f} 秒')
    print(f'下载速度：{file_size_MB / cost_time:0.2f}M/s')


def get_video_info():
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36'
        }

        url = "https://www.bilibili.com/video/BV17K4y1x7gs"
        response = requests.get(url=url, headers=headers)
        if response.status_code == 200:
            # bs = BeautifulSoup(response.text, 'html.parser')
            # 取视频标题
            # video_title = bs.find('span', class_='tit').get_text()

            # 取视频链接
            pattern = re.compile(r"<script>window\.__playinfo__=(.*?)</script>", re.MULTILINE | re.DOTALL)
            # script = bs.find("script", text=pattern)
            # result = pattern.search(script.next).group(1)
            content = response.content.decode("utf-8")
            with open("./content.html", "w") as fr:
                fr.write(content)
            # print(content)
            result = pattern.search(content).group(1)
            with open("./play_info.txt", "w") as fr:
                fr.write(result)

            temp = json.loads(result)
            # dash = temp['data']['dash']
            # with open("dash.txt", "w") as fr:
            #     fr.write(json.dumps(dash))

            video = temp['data']['dash']['video']
            audio = temp['data']['dash']['audio']
            print(len(video), len(audio))

            video_url = None
            audio_url = None
            # 取第一个视频链接
            for item in temp['data']['dash']['video']:
                if 'baseUrl' in item.keys():
                    video_url = item['baseUrl']
                    print(f"video_url: {video_url}")
                    break

            for item in temp['data']['dash']['audio']:
                if 'baseUrl' in item.keys():
                    audio_url = item['baseUrl']
                    print(f"audio_url: {audio_url}")
                    break

            return video_url, audio_url
            # return {
            #     'title': video_title,
            #     'url': video_url
            # }
    except requests.RequestException:
        print('视频链接错误，请重新更换')


def get_video_info2(bv):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36'
        }

        url = f"https://www.bilibili.com/video/{bv}"
        response = requests.get(url=url, headers=headers)
        if response.status_code == 200:
            # bs = BeautifulSoup(response.text, 'html.parser')
            # 取视频标题
            # video_title = bs.find('span', class_='tit').get_text()

            # 取视频链接
            pattern = re.compile(r"<script>window\.__playinfo__=(.*?)</script>", re.MULTILINE | re.DOTALL)
            # script = bs.find("script", text=pattern)
            # result = pattern.search(script.next).group(1)
            content = response.content.decode("utf-8")
            with open("./content.html", "w") as fr:
                fr.write(content)
            # print(content)
            result = pattern.search(content).group(1)
            with open("./play_info.txt", "w") as fr:
                fr.write(result)

            temp = json.loads(result)
            # dash = temp['data']['dash']
            # with open("dash.txt", "w") as fr:
            #     fr.write(json.dumps(dash))

            video = temp['data']['dash']['video']
            audio = temp['data']['dash']['audio']
            print(len(video), len(audio))

            video_url = None
            audio_url = None
            # 取第一个视频链接
            for item in temp['data']['dash']['video']:
                if 'baseUrl' in item.keys():
                    video_url = item['baseUrl']
                    print(f"video_url: {video_url}")
                    break

            for item in temp['data']['dash']['audio']:
                if 'baseUrl' in item.keys():
                    audio_url = item['baseUrl']
                    print(f"audio_url: {audio_url}")
                    break

            return video_url, audio_url
            # return {
            #     'title': video_title,
            #     'url': video_url
            # }
    except requests.RequestException:
        print('视频链接错误，请重新更换')


def download_video():
    url = "https://upos-sz-mirror08ct.bilivideo.com/upgcxcode/09/58/214205809/214205809_nb2-1-30032.m4s?e=ig8euxZM2rNcNbdlhoNvNC8BqJIzNbfqXBvEqxTEto8BTrNvN0GvT90W5JZMkX_YN0MvXg8gNEV4NC8xNEV4N03eN0B5tZlqNxTEto8BTrNvNeZVuJ10Kj_g2UB02J0mN0B5tZlqNCNEto8BTrNvNC7MTX502C8f2jmMQJ6mqF2fka1mqx6gqj0eN0B599M=&uipk=5&nbs=1&deadline=1675142981&gen=playurlv2&os=08ctbv&oi=717409460&trid=850fba46747d4160a39cef1702074778u&mid=0&platform=pc&upsig=a977d0884b3106284ba5eb889a6dc10a&uparams=e,uipk,nbs,deadline,gen,os,oi,trid,mid,platform&bvc=vod&nettype=0&orderid=0,3&buvid=3D282D46-9588-E75E-9351-2B1E55A8A32781652infoc&build=0&agrr=1&bw=34443&logo=80000000"
    filename = 'video.mp4'
    opener = urllib.request.build_opener()
    opener.addheaders = [('Origin', 'https://www.bilibili.com'),
                         ('Referer', "https://www.bilibili.com/video/BV17K4y1x7gs"),
                         ('User-Agent',
                          'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url=url, filename=filename)


def download_video1(url, filename):
    opener = urllib.request.build_opener()
    opener.addheaders = [('Origin', 'https://www.bilibili.com'),
                         ('Referer', "https://www.bilibili.com/video/BV17K4y1x7gs"),
                         ('User-Agent',
                          'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url=url, filename=filename)


def download_video2(bv, url, filename):
    opener = urllib.request.build_opener()
    opener.addheaders = [('Origin', 'https://www.bilibili.com'),
                         ('Referer', f"https://www.bilibili.com/video/{bv}"),
                         ('User-Agent',
                          'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url=url, filename=filename)


def download_video_audio():
    video_url, audio_url = get_video_info()
    download_video1(audio_url, "audio_tmp.m4s")
    download_video1(video_url, "video_tmp.m4s")
    os.system('ffmpeg -i video_tmp.m4s -i audio_tmp.m4s -vcodec copy -acodec copy video_new.mp4')
    os.remove("video_tmp.m4s")
    os.remove("audio_tmp.m4s")


def download_video_audio2(bv):
    video_url, audio_url = get_video_info2(bv)
    download_video2(bv, audio_url, "audio_tmp.m4s")
    download_video2(bv, video_url, "video_tmp.m4s")
    os.system('ffmpeg -i video_tmp.m4s -i audio_tmp.m4s -vcodec copy -acodec copy video.mp4')
    os.remove("video_tmp.m4s")
    os.remove("audio_tmp.m4s")


def download_mp4():
    url = "https://upos-sz-mirror08ct.bilivideo.com/upgcxcode/21/97/927229721/927229721-1-16.mp4?e=ig8euxZM2rNcNbRVhwdVhwdlhWdVhwdVhoNvNC8BqJIzNbfq9rVEuxTEnE8L5F6VnEsSTx0vkX8fqJeYTj_lta53NCM=&uipk=5&nbs=1&deadline=1675138938&gen=playurlv2&os=08ctbv&oi=1781826420&trid=c328ea0aa52b4f3abb0a519fe7a55deeh&mid=0&platform=html5&upsig=b3ebac8cfc5f127f260f1a730fb05290&uparams=e,uipk,nbs,deadline,gen,os,oi,trid,mid,platform&bvc=vod&nettype=0&bw=13434&logo=80000000"
    filename = 'video.mp4'
    opener = urllib.request.build_opener()
    opener.addheaders = [('Origin', 'https://www.bilibili.com'),
                         ('Referer', "https://www.bilibili.com/video/BV17K4y1x7gs"),
                         ('User-Agent',
                          'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url=url, filename=filename)


def debug():
    '''
    list: https://www.bilibili.com/video/BV1824y1D79P/?spm_id_from=333.880.my_history.page.click&vd_source=36491544991a804b0cb3b95631ceae14
    single: https://www.bilibili.com/video/BV1RR4y1Y7pm/?spm_id_from=333.880.my_history.page.click&vd_source=36491544991a804b0cb3b95631ceae14
    '''
    bv = "BV1824y1D79P"
    # get_by_bv(bv)
    # aid = "988992789"
    # cid = "927229721"
    # down_by_aid_cid(aid, cid)
    # get_video_info()
    # download_video()
    download_video_audio2(bv)
    print("done")


class BiliDown(object):
    BaseUrl = "https://www.bilibili.com"
    ViewUrl = "https://api.bilibili.com/x/web-interface/view?bvid="

    # PlayUrl = "https://api.bilibili.com/x/player/playurl?avid=" + aid + "&cid=" + cid + "&qn=80&type=mp4&platform=html5&high_quality=1"

    def __init__(self, bv):
        self.bv = bv

    def get_play_info(self):
        video_url = None
        audio_url = None
        video_title = None
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36'
            }

            url = f"https://www.bilibili.com/video/{self.bv}"
            response = requests.get(url=url, headers=headers)
            if response.status_code == 200:

                title_pattern = re.compile(r"class=\"video-title tit\">(.*?)</h1>", re.MULTILINE | re.DOTALL)
                info_pattern = re.compile(r"<script>window\.__playinfo__=(.*?)</script>", re.MULTILINE | re.DOTALL)
                content = response.content.decode("utf-8")
                # 取视频标题
                video_title = title_pattern.search(content).group(1)
                print(f"video_title: {video_title}")
                info_str = info_pattern.search(content).group(1)
                play_info = json.loads(info_str)

                video = play_info['data']['dash']['video']
                audio = play_info['data']['dash']['audio']
                print(len(video), len(audio))

                # 取第一个视频链接
                for item in play_info['data']['dash']['video']:
                    if 'baseUrl' in item.keys():
                        video_url = item['baseUrl']
                        print(f"video_url: {video_url}")
                        break

                for item in play_info['data']['dash']['audio']:
                    if 'baseUrl' in item.keys():
                        audio_url = item['baseUrl']
                        print(f"audio_url: {audio_url}")
                        break

                return video_url, audio_url, video_title
        except requests.RequestException:
            print('视频链接错误，请重新更换')
            return video_url, audio_url, video_title

    def download_file(self, url, filename):
        opener = urllib.request.build_opener()
        opener.addheaders = [('Origin', 'https://www.bilibili.com'),
                             ('Referer', f"https://www.bilibili.com/video/{self.bv}"),
                             ('User-Agent',
                              'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url=url, filename=filename)

    def download_mp4(self):
        video_url, audio_url, video_title = self.get_play_info()
        audio_filename = f"{video_title}_a.m4s"
        video_filename = f"{video_title}_v.m4s"
        filename = f"{video_title}.mp4"
        self.download_file(audio_url, audio_filename)
        self.download_file(video_url, video_filename)
        os.system(f"ffmpeg -i {video_filename} -i {audio_filename} -vcodec copy -acodec copy {filename}")
        os.remove(f"{video_filename}")
        os.remove(f"{audio_filename}")

    def get_aid_cid(self):
        url = f"https://api.bilibili.com/x/web-interface/view?bvid={self.bv}"
        response = requests.request("GET", url)
        aid = None
        cid = None
        if response.status_code == 200:
            content_str = response.content.decode("utf-8")
            content = json.loads(content_str)
            aid = content["data"]["aid"]
            cid = content["data"]["cid"]
        else:
            print("get_episodes error ......")

        return aid, cid

    def get_episodes(self):
        url = f"https://api.bilibili.com/x/web-interface/view?bvid={self.bv}"
        response = requests.request("GET", url)
        infos = []
        if response.status_code == 200:
            content_str = response.content.decode("utf-8")
            print(content_str)
            content = json.loads(content_str)
            # print(content)
            if "ugc_season" in content["data"]:
                ugc_season = content["data"]["ugc_season"]
                episodes = ugc_season["sections"][0]["episodes"]
                for item in episodes:
                    infos.append((item["bvid"], item["aid"], item["cid"]))
            else:
                print("no episodes in content")
        else:
            print("get_episodes error ......")

        return infos

    def get_play_url(self):
        aid, cid = self.get_aid_cid()
        return f"https://api.bilibili.com/x/player/playurl?avid={aid}&cid={cid}&qn=80&type=mp4&platform=html5&high_quality=1"

    def batch_download_mp4(self):
        infos = self.get_episodes()

        if len(infos) > 0:
            for bv, _, _ in infos:
                d = BiliDown(bv)
                d.download_mp4()
                time.sleep(5)
                print("=" * 72)


def main():
    bv = "BV1XM41127ZE"
    downloader = BiliDown(bv)
    downloader.batch_download_mp4()
    # play_url = downloader.get_play_url()
    # print(play_url)
    # downloader.download_mp4()
    # batch_download_mp4(bv)


if __name__ == '__main__':
    main()
