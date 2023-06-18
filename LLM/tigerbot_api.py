import requests


def main():
    API_KEY = "c97aadf7415102682faa7a151b19bfab4223a04c22dd6d649ef5751a6fd7725c"
    url = "https://api.tigerbot.com/bot-service/ai_service/gpt"

    headers = {
        'Authorization': 'Bearer ' + API_KEY
    }

    payload = {
        "text": "推荐系统的关键技术有哪些，请详细介绍一下",
        "modelVersion": "tigerbot-7b-sft"
    }

    response = requests.post(url, headers=headers, json=payload)
    print(response.text)
    # print(response.text["result"])


if __name__ == '__main__':
    main()
