import requests
import json
import re, time, datetime

from WebDriverCommon import SleepMode, WebDriverCommon

headers = {
    'accept': '*/*',
    'accept-language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
    'cache-control': 'no-cache',
    'content-type': 'application/json',
    'origin': 'https://www.tripadvisor.com',
    'pragma': 'no-cache',
    'priority': 'u=1, i',
    'referer': 'https://www.tripadvisor.com/Hotel_Review-g190327-d264936-Reviews-or10-1926_Le_Soleil_Hotel_Spa-Sliema_Island_of_Malta.html',
    'sec-ch-device-memory': '8',
    'sec-ch-ua': '"Not(A:Brand";v="99", "Google Chrome";v="133", "Chromium";v="133"',
    'sec-ch-ua-arch': '"x86"',
    'sec-ch-ua-full-version-list': '"Not(A:Brand";v="99.0.0.0", "Google Chrome";v="133.0.6943.98", "Chromium";v="133.0.6943.98"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-model': '""',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'same-origin',
    'sec-fetch-site': 'same-origin',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36',
    # 'cookie': 'TAUnique=%1%enc%3A0A8TZu9H5f%2B34Pa9RwAh1SPZDW3jOrUwVX3XvvslUWBfKyUX%2FbcDk%2BfECU%2FB93g8Nox8JbUSTxk%3D; TASameSite=1; TASSK=enc%3AAKx4pPQn1O2XTO%2BJvPvvpDkMhIazJcJGoHeGnnyKtZzPF5T94CCxHx8YJdflGf6%2F8B1oXhccufGmbg2iCWGV6AVnDSuWpkYmzQGYBdPKqGLWViCiBrC6VH%2BvMUaar87sYA%3D%3D; VRMCID=%1%V1*id.10568*llp.%2Fdevelopers*e.1739756824137; TATravelInfo=V2*A.2*MG.-1*HP.2*FL.3*RS.1; TATrkConsent=eyJvdXQiOiJTT0NJQUxfTUVESUEiLCJpbiI6IkFEVixBTkEsRlVOQ1RJT05BTCJ9; TADCID=9UK8mWI3R9FTIE25ABQCJ4S5rDsRRMescG99HippfoOq6gmQWRNkilQeokecxz8_FILkLM_N0dHA-8y47qF9cZMX0CvmctMZmJ4; ServerPool=B; PMC=V2*MS.88*MD.20250209*LD.20250216; TART=%1%enc%3A2EjFsvuh27D%2FuGfUw5IjExxHlrKHBhmzNunHuUF%2BfzKX%2B%2FbhJzBUuvL%2FYW7DHWCxQ4J%2BpECnWJ8%3D; TAReturnTo=%1%%2Fbusiness; G_AUTH2_MIGRATION=informational; TAAUTHEAT=pu7A_jqxi4L6WPIlABQC3I3IpT7XQpOMdOQw2Q027UOB68AG2JbdL173n3bCzo0ca0cHH2U--AM9PiwHZsc3w2TSlPOkrbtDOdmJZz4EasQN7_WJ21KMwxrl38pdHizJmxw4O_0DfKQxqy7l5R06mDNJcgregTuvOOhl0BFBkwO0Ly868sNtLPfD1fQ0u8FYj65zzHqzXBgQoSxZkp41mgJydTUSYTmU4kyO; AMZN-Token=v2FweIAwcXdkcU1UUWxYZDNzVTk4S1ZDZW11WklBSDgybmpEZEZxblFyQnEya0NPclJ4Wk93R3VyWElDR1VLRUFhbU1nQnpvYUlDcXB6U2JhWkNHUWtBWGtKdndCZEg1azFMNVZFbThodTVaQmFWSmZJNmo5eFdGQ09jN3RUc3hnUjA0bmJrdgFiaXZ4IFl5NDQ3Nys5NzcrOUxYRVg3Nys5NzcrOWJlKy92UT09/w==; TAUD=LA-1739752216927-1*RDD-1-2025_02_09*LG-1844774-2.1.F.*LD-1844775-.....; roybatty=TNI1625!AHcH5BtGaKyGosjI9SpL3EN%2Bq9cTJJCnrLD3tAdswfnlWZAjS2XfqX9Suw7sXgqt7%2BoIrubrV0k02PbC84YtxjvdF5%2BN41tnNss2GRL8iEvopURBkweJP7pB28Ra%2B5gSmccBkoyHx7IGLD2JkPF9VuUWOIpvEWdykJlu%2BzhsEkGTx1mieCwefMq5hizuYmFbNw%3D%3D%2C1; PAC=AEC_09F2-71tsFVBYDzBsOIUDUctMMOxkZ8R6SoEwoKUEKJYJVw5eugEp94s7leqz7MWeB6oSuttHAZf8TbqJg2kt2VBxzGZK6ZbmNRI5GCm6G-VHf3U0pJGpRJnybXtfPLGwWMde5LEfrdjCSfpNIVm4L0LC-1bXlEGBQX86OjjHJB3F5TpqMAgg1l8fZVqX1LphuM-cQDHmjRQkQI0xl4vlPP9vxd8I1TtGloxAkmKSNn8ShQSMetGICdl0ZUamDh96UO8FiVwBmajJO-H62g%3D; datadome=RWGeP2XLBiCuAtdvRFJWIxNsXUWftY76lBN3fnQI32BpFjFmeRWzZQmM9oQ~7nRVG27a_2UMsP7XgRTjSRpjaHnyFNyjmhBfYAPTaGdOUB~CWz4hbScMgP6skLY53_Bo; OptanonConsent=isGpcEnabled=0&datestamp=Mon+Feb+17+2025+10%3A05%3A58+GMT%2B0800+(%E5%8F%B0%E5%8C%97%E6%A8%99%E6%BA%96%E6%99%82%E9%96%93)&version=202405.2.0&browserGpcFlag=0&isIABGlobal=false&hosts=&consentId=2C2C3B4C163EA2B34530A0FC114B6BF1&interactionCount=1&isAnonUser=1&landingPath=NotLandingPage&groups=C0001%3A1%2CC0002%3A1%2CC0003%3A1%2CC0004%3A1&AwaitingReconsent=false; TASession=V2ID.B50D4E635BDCFB4DFDDBDE9C9FB17273*SQ.14*LS.Hotel_Review*HS.recommended*ES.popularity*DS.5*SAS.popularity*FPS.oldFirst*TS.2C2C3B4C163EA2B34530A0FC114B6BF1*FA.1*DF.0*TRA.true*EAU.Y; __vt=9NA6f_Iw48_bT9C4ABQCT24E-H_BQo6gx1APGQJPtzOJD20vsmHC6OBBXZ0XalWbI-PpRRfpvEuwU0OwAQIwMbfYpUFE4U1qWQ82M8znRubmVlXdWY3LU5muCKL3Wf4jh_HUhHh1qKwEBOSofgovloZOsA9n1NKb-GbyZ7yvFXEmiFeU9-sAL4pDt_qjakTbYPPmZdMGgboHpZoFGZhihNvI6xgwuuq6KfdK84DtPCBDnQpUZd7Z2WnXJRiFSk4naIpjHLYeci8hXmahLhS_A0yX3FrazdEe-4uDi1HP16uBnfao8eus2A',
}

# lang: en

class TripAdvisorRequestCrawler:
    def __init__(self, preRegisteredQueryId: str):

        self.headers = {
            'accept': '*/*',
            'accept-language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
            'cache-control': 'no-cache',
            'content-type': 'application/json',
            'origin': 'https://www.tripadvisor.com',
            'pragma': 'no-cache',
            'priority': 'u=1, i',
            'referer': 'https://www.tripadvisor.com/Hotel_Review-g190327-d264936-Reviews-or10-1926_Le_Soleil_Hotel_Spa-Sliema_Island_of_Malta.html',
            'sec-ch-device-memory': '8',
            'sec-ch-ua': '"Not(A:Brand";v="99", "Google Chrome";v="133", "Chromium";v="133"',
            'sec-ch-ua-arch': '"x86"',
            'sec-ch-ua-full-version-list': '"Not(A:Brand";v="99.0.0.0", "Google Chrome";v="133.0.6943.98", "Chromium";v="133.0.6943.98"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-model': '""',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'same-origin',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36',
            # 'cookie': 'TAUnique=%1%enc%3A0A8TZu9H5f%2B34Pa9RwAh1SPZDW3jOrUwVX3XvvslUWBfKyUX%2FbcDk%2BfECU%2FB93g8Nox8JbUSTxk%3D; TASameSite=1; TASSK=enc%3AAKx4pPQn1O2XTO%2BJvPvvpDkMhIazJcJGoHeGnnyKtZzPF5T94CCxHx8YJdflGf6%2F8B1oXhccufGmbg2iCWGV6AVnDSuWpkYmzQGYBdPKqGLWViCiBrC6VH%2BvMUaar87sYA%3D%3D; VRMCID=%1%V1*id.10568*llp.%2Fdevelopers*e.1739756824137; TATravelInfo=V2*A.2*MG.-1*HP.2*FL.3*RS.1; TATrkConsent=eyJvdXQiOiJTT0NJQUxfTUVESUEiLCJpbiI6IkFEVixBTkEsRlVOQ1RJT05BTCJ9; TADCID=9UK8mWI3R9FTIE25ABQCJ4S5rDsRRMescG99HippfoOq6gmQWRNkilQeokecxz8_FILkLM_N0dHA-8y47qF9cZMX0CvmctMZmJ4; ServerPool=B; PMC=V2*MS.88*MD.20250209*LD.20250216; TART=%1%enc%3A2EjFsvuh27D%2FuGfUw5IjExxHlrKHBhmzNunHuUF%2BfzKX%2B%2FbhJzBUuvL%2FYW7DHWCxQ4J%2BpECnWJ8%3D; TAReturnTo=%1%%2Fbusiness; G_AUTH2_MIGRATION=informational; TAAUTHEAT=pu7A_jqxi4L6WPIlABQC3I3IpT7XQpOMdOQw2Q027UOB68AG2JbdL173n3bCzo0ca0cHH2U--AM9PiwHZsc3w2TSlPOkrbtDOdmJZz4EasQN7_WJ21KMwxrl38pdHizJmxw4O_0DfKQxqy7l5R06mDNJcgregTuvOOhl0BFBkwO0Ly868sNtLPfD1fQ0u8FYj65zzHqzXBgQoSxZkp41mgJydTUSYTmU4kyO; AMZN-Token=v2FweIAwcXdkcU1UUWxYZDNzVTk4S1ZDZW11WklBSDgybmpEZEZxblFyQnEya0NPclJ4Wk93R3VyWElDR1VLRUFhbU1nQnpvYUlDcXB6U2JhWkNHUWtBWGtKdndCZEg1azFMNVZFbThodTVaQmFWSmZJNmo5eFdGQ09jN3RUc3hnUjA0bmJrdgFiaXZ4IFl5NDQ3Nys5NzcrOUxYRVg3Nys5NzcrOWJlKy92UT09/w==; TAUD=LA-1739752216927-1*RDD-1-2025_02_09*LG-1844774-2.1.F.*LD-1844775-.....; roybatty=TNI1625!AHcH5BtGaKyGosjI9SpL3EN%2Bq9cTJJCnrLD3tAdswfnlWZAjS2XfqX9Suw7sXgqt7%2BoIrubrV0k02PbC84YtxjvdF5%2BN41tnNss2GRL8iEvopURBkweJP7pB28Ra%2B5gSmccBkoyHx7IGLD2JkPF9VuUWOIpvEWdykJlu%2BzhsEkGTx1mieCwefMq5hizuYmFbNw%3D%3D%2C1; PAC=AEC_09F2-71tsFVBYDzBsOIUDUctMMOxkZ8R6SoEwoKUEKJYJVw5eugEp94s7leqz7MWeB6oSuttHAZf8TbqJg2kt2VBxzGZK6ZbmNRI5GCm6G-VHf3U0pJGpRJnybXtfPLGwWMde5LEfrdjCSfpNIVm4L0LC-1bXlEGBQX86OjjHJB3F5TpqMAgg1l8fZVqX1LphuM-cQDHmjRQkQI0xl4vlPP9vxd8I1TtGloxAkmKSNn8ShQSMetGICdl0ZUamDh96UO8FiVwBmajJO-H62g%3D; datadome=RWGeP2XLBiCuAtdvRFJWIxNsXUWftY76lBN3fnQI32BpFjFmeRWzZQmM9oQ~7nRVG27a_2UMsP7XgRTjSRpjaHnyFNyjmhBfYAPTaGdOUB~CWz4hbScMgP6skLY53_Bo; OptanonConsent=isGpcEnabled=0&datestamp=Mon+Feb+17+2025+10%3A05%3A58+GMT%2B0800+(%E5%8F%B0%E5%8C%97%E6%A8%99%E6%BA%96%E6%99%82%E9%96%93)&version=202405.2.0&browserGpcFlag=0&isIABGlobal=false&hosts=&consentId=2C2C3B4C163EA2B34530A0FC114B6BF1&interactionCount=1&isAnonUser=1&landingPath=NotLandingPage&groups=C0001%3A1%2CC0002%3A1%2CC0003%3A1%2CC0004%3A1&AwaitingReconsent=false; TASession=V2ID.B50D4E635BDCFB4DFDDBDE9C9FB17273*SQ.14*LS.Hotel_Review*HS.recommended*ES.popularity*DS.5*SAS.popularity*FPS.oldFirst*TS.2C2C3B4C163EA2B34530A0FC114B6BF1*FA.1*DF.0*TRA.true*EAU.Y; __vt=9NA6f_Iw48_bT9C4ABQCT24E-H_BQo6gx1APGQJPtzOJD20vsmHC6OBBXZ0XalWbI-PpRRfpvEuwU0OwAQIwMbfYpUFE4U1qWQ82M8znRubmVlXdWY3LU5muCKL3Wf4jh_HUhHh1qKwEBOSofgovloZOsA9n1NKb-GbyZ7yvFXEmiFeU9-sAL4pDt_qjakTbYPPmZdMGgboHpZoFGZhihNvI6xgwuuq6KfdK84DtPCBDnQpUZd7Z2WnXJRiFSk4naIpjHLYeci8hXmahLhS_A0yX3FrazdEe-4uDi1HP16uBnfao8eus2A',
        }

        self.preRegisteredQueryId = preRegisteredQueryId

    def save_Hotel_reviews_count(self, hotelId: int, selectionsLang: str, langCode: str, savedFilePath: str):
        json_data = [
            {
                'variables': {
                    'hotelId': hotelId,
                    'filters': [
                        {
                            'axis': 'LANGUAGE',
                            'selections': [
                                selectionsLang,
                            ],
                        },
                    ],
                    'limit': 10,
                    'offset': 0,
                    'sortType': None,
                    'sortBy': 'SERVER_DETERMINED',
                    'language': langCode,
                },
                'extensions': {
                    'preRegisteredQueryId': self.preRegisteredQueryId,
                },
            },
        ]

        response = requests.post(
            'https://www.tripadvisor.com/data/graphql/ids', 
            headers=self.headers, 
            json=json_data)
        
        with open(savedFilePath, 'w', encoding='utf8') as sw:
            sw.write(json.dumps(json.loads(response.text), indent=2, ensure_ascii=False))

def collect_Hotel_reviews(hotelId: int, offset: int, preRegisteredQueryId: str):
    pass

def save_review_first_page(preRegisteredQueryId: str):

    crawler = TripAdvisorRequestCrawler(preRegisteredQueryId)

    with open("./Taiwan-Hotels.txt", "r", encoding='utf8') as sr:
        lineNumber = 0

        for line in sr:
            lineNumber += 1

            columns = line.split("\t")
            hotel_name = columns[0]
            url = columns[1]

            # 使用正則表達式找出 d 後面的數字
            match = re.search(r"-d(\d+)-Reviews", url)

            hotel_id = int(match.group(1))
            print("[{}] {}: {}".format(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"), hotel_name, hotel_id))


            # crawler.save_Hotel_reviews_count(hotel_id, "zhTW", "zh", "./output/en_reviews/reviews_en_{0}_01.json".format(hotel_id))
            crawler.save_Hotel_reviews_count(hotel_id, "en", "en", "./output/en_reviews/reviews_en_{0}_01.json".format(hotel_id))


            WebDriverCommon.sleep_strategy(SleepMode.Quick.value)

            if lineNumber % 10 == 0:
                WebDriverCommon.sleep_strategy(SleepMode.Long.value)




if __name__ == '__main__':
    
    # crawler = TripAdvisorRequestCrawler("b8bc339a61f4ea2e")
    # crawler.save_Hotel_reviews_count(302103, "zhTW", "zh", "./output/reviews_zh_302103_01.json")
    save_review_first_page("b8bc339a61f4ea2e")
