# -*- coding: utf-8 -*-
import requests
import json
import time
import re
import urllib
import os
from urllib.request import urlretrieve


class BiliDown(object):

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
        return f"https://api.bilibili.com/x/player/playurl?avid={aid}&cid={cid}" \
               f"&qn=80&type=mp4&platform=html5&high_quality=1"

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
