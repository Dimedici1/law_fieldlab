from bs4 import BeautifulSoup
import warnings
import random
import requests
warnings.filterwarnings("ignore")


# Functions

def compact(lst):
    """
    This function removes all None or null values from the provided list.

    :param lst: List that may contain null or None values.
    :returns: A list with null or None values removed.
    """
    return list(filter(None, lst))


def rand_useragent():
    """
    This function generates a random set of request headers including user agent, engine, language, encoding, and accept.
    These headers mimic the behavior of different types of browsers to avoid detection during web scraping.

    :returns: tuple of 5 elements - user_agent, engine, accept_language, accept_encoding, accept.
    """
    accept_list = ['text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                   'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                   'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                   'text/html,application/xhtml+xml,application/xml;q=0.7,image/webp,*/*;q=0.8',
                   'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                   'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                   'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                   'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.5'
                   ]

    accept_encoding = ['gzip', 'deflate']

    accept_language_list = ["en-US,en;q=0.9,es;q=0.8", "en-GB,en;q=0.5,es;q=0.6", "fr-FR,en;q=0.1,es;q=0.5",
                            "fr-FR,en;q=0.1,es;q=0.5", "fr-FR,en;q=0.1,es;q=0.5", "fr-CH,es;q=1.1,es;q=0.7",
                            "de-CH,fr;q=0.7,es;q=0.8", "es-ES,es;q=0.1,es;q=0.5"]

    search_engine_list = ["https://www.yahoo.com", "https://www.duckduckgo.com", "https://www.google.com",
                          "https://www.bing.com", "https://www.baidu.com", "https://www.ask.com",
                          "https://www.yandex.com", "https://www.opera.com"]

    user_agent_list = [
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.1',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:77.0) Gecko/20100101 Firefox/77.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97',
        'Mozilla/5.0 (Linux; Android 7.1.1; SM-J510FN Build/NMF26X; wv) AppleWebKit/537.36 (KHTML, like Gecko)',
        'Mozilla/5.0 (Linux; Android 11; M2007J20CG) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.41',
        'Mozilla/5.0 (Linux; Android 9; SM-A207M Build/PPR1.180610.011; wv) AppleWebKit/537.36 (KHTML, like Gecko)']

    # Choose randomly from lists
    user_agent = random.choice(user_agent_list)
    engine = random.choice(search_engine_list)
    acc_lang = random.choice(accept_language_list)
    acc_encode = random.choice(accept_encoding)
    accept = random.choice(accept_list)
    return user_agent, engine, acc_lang, acc_encode, accept


def create_soup(url_1, max_retries=10):
    """
    This function tries to send a request to the given URL and parse the page content with BeautifulSoup.
    If the request fails, it retries until the maximum number of retries is reached.

    :param url_1: The URL to send a GET request to.
    :param max_retries: The maximum number of times to retry sending a request in case of failure.
    :returns: BeautifulSoup object containing the HTML content of the page or 'nan' if max retries reached with failure.
    """
    ua, engine, acc_lang, acc_encode, accept = rand_useragent()
    page = None
    counter = 0

    # Continue requesting until connection successful or max number of tries is reached
    while page is None and counter <= max_retries:
        try:
            page = requests.get(url_1, headers={
                'User-Agent': ua,
                "Referer": engine,
                "Accept": accept,
                "Accept-Encoding": acc_encode,
                "Accept-Language": acc_lang
            })
        # If the url passed is nan then a nan value should be returned. Otherwise retry.
        except:
            print('connection lost', url_1)
            counter += 1
        if counter == max_retries:
            return 'nan'
    # Otherwise create a beautiful soup object
    soup_1 = BeautifulSoup(page.content, 'html.parser')
    return soup_1
