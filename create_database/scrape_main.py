from scraping_functions import create_soup
import re
import time
import random

warnings.filterwarnings('ignore')

def format_text(text):
    # Remove leading and trailing whitespaces
    text = text.strip()

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    # Replace '—' followed by spaces with a bullet point and a newline
    text = re.sub(r'\s*—\s*', '', text)

    # Insert a newline before each major section
    #for section in ['SUMMARY', 'KEY POINTS', 'BACKGROUND', 'ACT', 'RELATED ACTS']:
    #    text = text.replace(section, f'\n\n{section}')

    # Use regular expression to remove the ending phrase with varying dates
    text = re.sub(r'last update \d{2}\.\d{2}\.\d{4}Top', '', text)

    return text


def scrape_summaries(summary_links):
    summaries_list = []
    valid_summary_links = []
    for link in summary_links:
        print(link)
        n = 0
        doc = None
        while doc is None and n < 10:
            # Pause for a random time between 0 and 1 seconds
            time_to_sleep = random.uniform(0, 1)
            time.sleep(time_to_sleep)
            # Create soup object
            soup = create_soup(link)
            # Get list of all rows
            doc = str(soup.find('div', {'id': 'document1'}))
            if doc == "None":
                print("Summary not found")
                doc = None
                if n == 9:
                    break
            n += 1

        if doc is not None:
            # Remove all <>
            cleaned_text = re.sub(r'<.*?>', '', doc)
            # Reformat summaries
            cleaned_text = format_text(cleaned_text)
            # Append to list
            summaries_list.append(cleaned_text)
            valid_summary_links.append(link)
        else:
            print(f"Failed to scrape summary for {link}")

    return summaries_list, valid_summary_links

def scrape_full_documents(urls):
    documents_list = list()
    summary_links = list()
    for url in urls:
        # Pause for a random time between 0 and 1 seconds
        time_to_sleep = random.uniform(0, 1)
        time.sleep(time_to_sleep)
        print(url)
        # Create soup object
        n = 0
        doc = None
        while doc is None and n < 30:
            soup = create_soup(url)
            # Get list of all rows
            doc = str(soup.find('div', {'class': 'tabContent'}, {'id': 'document1'}))
            if doc == "None":
                doc = None
                print("Document not found")
            n += 1
        # Remove all <>
        cleaned_text = re.sub(r'<.*?>', '', doc)
        # Now remove all newline characters
        #cleaned_text = re.sub(r'\n', '', cleaned_text)
        # Reformat text
        cleaned_text = format_text(cleaned_text)
        # Append to list
        documents_list.append(cleaned_text)

        # Get summary link
        links = soup.find('nav', {'id': 'AffixSidebar'}).find('ul', {'class': 'MenuList'}).find_all('li')

        # Iterate through them to find the one with the 'legissumTab' class
        for li in links:
            if 'legissumTab' in li.get('class', []):
                link = li.a.get('href')
                summary_links.append("https://eur-lex.europa.eu/" + link)
                break
            else:
                pass

    # Scrape summaries
    summaries_list, summary_links = scrape_summaries(summary_links)

    # Combine both lists
    #documents_list = documents_list + summaries_list

    return documents_list, summaries_list, summary_links
